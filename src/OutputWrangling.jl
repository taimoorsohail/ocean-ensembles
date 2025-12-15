module OutputWrangling

using Oceananigans
using Oceananigans.Fields: location
using JLD2
using ClimaOcean
using Glob
using Printf

export combine_ranks, identify_combination_targets, create_grid

function grid_metrics(prefix, ranks)
    file   = jldopen(prefix * "_rank$(ranks[1]).jld2")
    data   = file["grid/underlying_grid"]
    Nx, Ny, Nz = data["Nx"], data["Ny"]*Integer(length(ranks)), data["Nz"]
    Hx, Hy, Hz = data["Hx"], data["Hy"], data["Hz"]
    Lz = data["Lz"]

    nx = Integer(Nx / length(ranks))
    ny = Integer(Ny / length(ranks))

    depth = -Lz # Depth of the ocean in meters
    z_faces = ExponentialDiscretization(Nz, depth, 0)
    return Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, Lz, z_faces
end

function create_grid(prefix, ranks; gridtype = "TripolarGrid")
    Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, Lz, z_faces = grid_metrics(prefix, ranks)
    if gridtype == "LatitudeLongitudeGrid"
        grid = LatitudeLongitudeGrid(CPU();
                                     size = (Nx, Ny, Nz),
                                     z = z_faces,
                                     halo = (Hx, Hy, Hz),
                                     latitude  = (-75, 75),
                                     longitude = (0, 360))        
    elseif gridtype == "TripolarGrid"
        grid = TripolarGrid(CPU();
                            size = (Nx, Ny, Nz),
                            z = z_faces,
                            halo = (Hx, Hy, Hz),
                            first_pole_longitude = 70,
                            north_poles_latitude = 55)
    end

    bottom_height = read_bathymetry(prefix, ranks)

    grid  = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))
    return grid
end

function read_bathymetry(prefix, ranks)
    Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, Lz, z_faces = grid_metrics(prefix, ranks)

    bottom_height = zeros(Nx, Ny)

    for rank in ranks
        irange = ny * rank + 1 : ny * (rank + 1)
        file   = jldopen(prefix * "_rank$(rank).jld2")
        data   = file["grid/immersed_boundary/bottom_height"][Hx+1:Nx+Hx, Hy+1:ny+Hy,  1]
        bottom_height[:, irange] .= data
        close(file)
    end

    return bottom_height
end

function combine_ranks(prefix, grid)
    iter_rank_map = identify_combination_targets(basename(prefix), dirname(prefix))
    iterations    = collect(keys(iter_rank_map))

    for iteration in iterations
        ranks = iter_rank_map[iteration]
        outpath = prefix * "_iteration$(iteration).jld2"

        # --------------------------------------------------------------
        # SKIP IF OUTPUT FILE ALREADY EXISTS
        # --------------------------------------------------------------
        if isfile(outpath)
            @info "Skipping iteration $iteration because output already exists: $outpath"
            continue
        end

        # --------------------------------------------------------------
        # Probe first rank for timestep metadata
        # --------------------------------------------------------------
        file0 = jldopen(prefix * "_iteration$(iteration)_rank$(ranks[1]).jld2", "r")

        if !haskey(file0, "timeseries/t")
            @warn "Skipping iteration $iteration: key 'timeseries/t' not found."
            close(file0)
            continue
        end

        tkeys = collect(keys(file0["timeseries/t"]))       # e.g., ["0", "120"]
        iters = sort(parse.(Int, tkeys))                   # numeric sort
        times = Float64[file0["timeseries/t/$(k)"] for k in iters]

        close(file0)
        @info "Combining ranks $(ranks) for iteration $(iteration)"

        # --------------------------------------------------------------
        # Create output FieldTimeSeries (one file, 6 variables)
        # --------------------------------------------------------------
        utmp = FieldTimeSeries{Face,   Center, Nothing}(grid, times; backend=OnDisk(), path=outpath, name="u")
        vtmp = FieldTimeSeries{Center, Face,   Nothing}(grid, times; backend=OnDisk(), path=outpath, name="v")
        wtmp = FieldTimeSeries{Center, Face,   Nothing}(grid, times; backend=OnDisk(), path=outpath, name="w")
        Ttmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=outpath, name="T")
        Stmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=outpath, name="S")
        etmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=outpath, name="e")

        # --------------------------------------------------------------
        # Worker: stitch ranks into a global field (FAST)
        # --------------------------------------------------------------
        function stitch!(fts, prefix, ranks, iteration, iters, grid)
            Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, Lz, z_faces =
                grid_metrics(prefix * "_iteration$(iteration)", ranks)

            field = Field{location(fts)...}(grid)

            # open ALL rank files ONCE
            rfiles = Dict(r => jldopen(prefix * "_iteration$(iteration)_rank$(r).jld2", "r")
                          for r in ranks)

            try
                for (idx, tkey) in enumerate(iters)

                    for r in ranks
                        f = rfiles[r]
                        arr = f["timeseries/$(fts.name)/$(tkey)"][:, :, :]
                        irange = ny * r + 1 : ny * (r + 1)
                        interior(field, :, irange, :) .= arr
                    end

                    set!(fts, field, idx)
                end

            finally
                foreach(close, values(rfiles))
            end
        end

        # --------------------------------------------------------------
        # Build each variable (minimising open files and allocations)
        # --------------------------------------------------------------
        stitch!(utmp, prefix, ranks, iteration, iters, grid)
        stitch!(vtmp, prefix, ranks, iteration, iters, grid)
        stitch!(wtmp, prefix, ranks, iteration, iters, grid)
        stitch!(Ttmp, prefix, ranks, iteration, iters, grid)
        stitch!(Stmp, prefix, ranks, iteration, iters, grid)
        stitch!(etmp, prefix, ranks, iteration, iters, grid)

        @info "Finished writing iteration $(iteration) → $outpath"

        # --------------------------------------------------------------
        # CRITICAL: Explicitly close all output JLD2 file handles
        # --------------------------------------------------------------
        close(utmp.output)
        close(vtmp.output)
        close(wtmp.output)
        close(Ttmp.output)
        close(Stmp.output)
        close(etmp.output)

        # --------------------------------------------------------------
        # Release MMAP buffers (prevents SystemError: msync errors)
        # --------------------------------------------------------------
        GC.gc()
    end

    return nothing
end


function identify_combination_targets(prefix, output_path; type = "iterrank")
    if type == "iterrank"
        file_pattern = prefix * "_iteration*_rank*"
        files = glob(file_pattern, output_path)
        pattern = Regex("^" * prefix * "_iteration(\\d+)_rank(\\d+)\\.jld2")
        iter_rank_map = Dict{Int, Vector{Int}}()

        for file in files
            fname = basename(file)
            m = match(pattern, fname)
            if m !== nothing
                iter = parse(Int, m.captures[1])
                rank = parse(Int, m.captures[2])
                push!(get!(iter_rank_map, iter, Int[]), rank)
            end
        end
        return iter_rank_map

    elseif type == "iter"
        @show file_pattern = prefix * "_iteration*"
        @show files = glob(file_pattern, output_path)
        @show pattern = Regex("^" * prefix * "_iteration(\\d+)\\.jld2")

        iter_map = Dict{Int, Vector{Int}}()

        for file in files
            fname = basename(file)
            m = match(pattern, fname)
            if m !== nothing
                iter = parse(Int, m.captures[1])
                push!(get!(iter_map, iter, Int[]), 0)  # use dummy rank 0
            end
        end
        return iter_map
    end
end 

function combine_iters(prefix, prefix_out; remove_split_files = false)
    iter_rank_map = identify_combination_targets(basename(prefix), dirname(prefix); type = "iter")
    iterations = sort(collect(keys(iter_rank_map)))

    combined = Dict{String, Any}()

    for iteration in iterations
        filename = prefix * "_iteration$(iteration).jld2"
        println("Reading $filename")

        jldopen(filename, "r") do file
            for key in keys(file["timeseries"])
                data = file["timeseries/$(key)"]

                if haskey(combined, key)
                    combined[key] = cat(combined[key], data; dims=1)
                else
                    combined[key] = data
                end
            end
        end
    end

    # Save combined data
    outfile = prefix_out * ".jld2"
    println("Saving combined file to $outfile")

    jldsave(outfile; timeseries=combined)

    # # Optionally remove originals
    # if remove_split_files
    #     for iteration in iterations
    #         rm(prefix * "_iteration$(iteration).jld2")
    #     end
    # end
end

end