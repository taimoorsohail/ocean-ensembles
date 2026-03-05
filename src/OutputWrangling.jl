module OutputWrangling

using Oceananigans
using Oceananigans.Fields: location
using JLD2
using NumericalEarth
using Glob
using Printf

export combine_ranks, identify_combination_targets, create_grid, read_bathymetry

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

function grid_metrics(prefix)
    file   = jldopen(prefix* ".jld2")
    data   = file["grid/underlying_grid"]
    Nx, Ny, Nz = data["Nx"], data["Ny"], data["Nz"]
    Hx, Hy, Hz = data["Hx"], data["Hy"], data["Hz"]
    Lz = data["Lz"]

    depth = -Lz # Depth of the ocean in meters
    z_faces = ExponentialDiscretization(Nz, depth, 0)
    return Nx, Ny, Nz, Hx, Hy, Hz, Lz, z_faces
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

function create_grid(prefix; gridtype = "TripolarGrid")
    Nx, Ny, Nz, Hx, Hy, Hz, Lz, z_faces = grid_metrics(prefix)
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

    bottom_height = read_bathymetry(prefix)

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

function read_bathymetry(prefix)
    Nx, Ny, Nz, Hx, Hy, Hz, Lz, z_faces = grid_metrics(prefix)

    bottom_height = zeros(Nx, Ny)

    file   = jldopen(prefix * ".jld2")
    data   = file["grid/immersed_boundary/bottom_height"][Hx+1:Nx+Hx, Hy+1:Ny+Hy,  1]
    bottom_height .= data
    close(file)

    return bottom_height
end


function combine_ranks(prefix, grid; iterrun = "run")
    iter_rank_map = identify_combination_targets(basename(prefix), dirname(prefix); iterrun = iterrun)
    iterations    = sort(collect(keys(iter_rank_map)))

    for iteration in iterations
        run = lpad(string(iteration), 4, '0')

        ranks = iter_rank_map[iteration]
        outpath = prefix * "_$(iterrun)$(run).jld2"

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
        file0 = jldopen(prefix * "_$(iterrun)$(run)_rank$(ranks[1]).jld2", "r")

        if !haskey(file0, "timeseries/t")
            @warn "Skipping iteration $iteration: key 'timeseries/t' not found."
            close(file0)
            continue
        end
        vars = keys(file0["timeseries"])
        tkeys = collect(keys(file0["timeseries/t"]))       # e.g., ["0", "120"]
        iters = sort(parse.(Int, tkeys))                   # numeric sort
        times = Float64[file0["timeseries/t/$(k)"] for k in iters]

        @info "Combining ranks $(ranks) for iteration $(run)"

        # --------------------------------------------------------------
        # Create output FieldTimeSeries (one file, 6 variables)
        # --------------------------------------------------------------
        tmp = []
        for var in vars
            @show var
            if var != "t"
                @info "Creating $(var) output FieldTimeSeries at $outpath"
                rawlocs = file0["timeseries/$(var)/serialized/location"]
                tmp = FieldTimeSeries{rawlocs[1],   rawlocs[2], Nothing}(grid, times; backend=OnDisk(), path=outpath, name=string(var))
                @info "Stitching $(var) for iteration $(run) → $outpath"
                @time set_distributed_field_time_series!(tmp, prefix, ranks, iteration, iters, grid)
            end
        end

        close(file0)

        # @info "Creating v output FieldTimeSeries at $outpath"
        # @time vtmp = FieldTimeSeries{Center, Face,   Nothing}(grid, times; backend=OnDisk(), path=outpath, name="v")
        # @info "Creating w output FieldTimeSeries at $outpath"
        # @time wtmp = FieldTimeSeries{Center, Face,   Nothing}(grid, times; backend=OnDisk(), path=outpath, name="w")
        # @info "Creating T output FieldTimeSeries at $outpath"
        # @time Ttmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=outpath, name="T")
        # @info "Creating S output FieldTimeSeries at $outpath"
        # @time Stmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=outpath, name="S")
        # @info "Creating e output FieldTimeSeries at $outpath"
        # @time etmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=outpath, name="e")

        # --------------------------------------------------------------
        # Worker: stitch ranks into a global field (FAST)
        # --------------------------------------------------------------
        # function stitch!(fts, prefix, ranks, iteration, iters, grid)
        #     Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, Lz, z_faces =
        #         grid_metrics(prefix * "_run$(run)", ranks)

        #     field = Field{location(fts)...}(grid)

        #     # open ALL rank files ONCE
        #     rfiles = Dict(r => jldopen(prefix * "_run$(run)_rank$(r).jld2", "r")
        #                   for r in ranks)

        #     try
        #         for (idx, tkey) in enumerate(iters)

        #             for r in ranks
        #                 f = rfiles[r]
        #                 arr = f["timeseries/$(fts.name)/$(tkey)"][:, :, :]
        #                 irange = ny * r + 1 : ny * (r + 1)
        #                 interior(field, :, irange, :) .= arr
        #             end

        #             set!(fts, field, idx)
        #         end

        #     finally
        #         foreach(close, values(rfiles))
        #     end
        # end

        # --------------------------------------------------------------
        # Build each variable (minimising open files and allocations)
        # --------------------------------------------------------------
        # @info "Stitching v for iteration $(run) → $outpath"
        # @time set_distributed_field_time_series!(vtmp, prefix, ranks, iteration, iters, grid)
        # @info "Stitching w for iteration $(run) → $outpath"
        # @time set_distributed_field_time_series!(wtmp, prefix, ranks, iteration, iters, grid)
        # @info "Stitching T for iteration $(run) → $outpath"
        # @time set_distributed_field_time_series!(Ttmp, prefix, ranks, iteration, iters, grid)
        # @info "Stitching S for iteration $(run) → $outpath"
        # @time set_distributed_field_time_series!(Stmp, prefix, ranks, iteration, iters, grid)
        # @info "Stitching e for iteration $(run) → $outpath"
        # @time set_distributed_field_time_series!(etmp, prefix, ranks, iteration, iters, grid)

        @info "Finished writing iteration $(run) → $outpath"

        # --------------------------------------------------------------
        # CRITICAL: Explicitly close all output JLD2 file handles
        # --------------------------------------------------------------
        @info "Closing output files for iteration $(run) → $outpath"
        # close(utmp.output)
        # close(vtmp.output)
        # close(wtmp.output)
        # close(Ttmp.output)
        # close(Stmp.output)
        # close(etmp.output)

        # --------------------------------------------------------------
        # Release MMAP buffers (prevents SystemError: msync errors)
        # --------------------------------------------------------------
        @info "Releasing MMAP buffers for iteration $(run) → $outpath"
        GC.gc()
    end

    return nothing
end

function set_distributed_field_time_series!(fts, prefix, ranks, iteration, iters, grid; iterrun = "run")
    run = lpad(string(iteration), 4, '0')
    Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, Lz, z_faces = grid_metrics(prefix * "_$(iterrun)$(run)", ranks) 
    field = Field{location(fts)...}(grid) 
    Ny = size(fts, 2) # loop over timesteps FIRST 
    for (idx, iter) in enumerate(iters) # fresh field for this timestep (critical!) 
        field = Field{location(fts)...}(grid) # fill the global domain rank-by-rank 
        for rank in ranks 
            file = jldopen(prefix * "_$(iterrun)$(run)_rank$(rank).jld2") # shape typically (Nx_local, Ny_local, Nz_local) 
            data = file["timeseries/$(fts.name)/$(iter)"][:, :, :] # y-range for rank 
            irange = ny * rank + 1 : ny * (rank + 1) # fill full vertical column (use ":" in last dim) 
            interior(field, :, irange, :) .= data 
            close(file)
        end # now write this full timestep to disk 
        set!(fts, field, idx) 
    end 
end

function identify_combination_targets(prefix, output_path; type = "iterrank", iterrun = "run")
    if type == "iterrank"
        file_pattern = prefix * "_$(iterrun)*_rank*"
        files = glob(file_pattern, output_path)
        pattern = Regex("^" * prefix * "_$(iterrun)(\\d+)_rank(\\d+)\\.jld2")
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
        file_pattern = prefix * "_$(iterrun)*"
        files = glob(file_pattern, output_path)
        pattern = Regex("^" * prefix * "_$(iterrun)(\\d+)\\.jld2")

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

function combine_iters(prefix, prefix_out; remove_split_files = false,  iterrun = "run")
    iter_rank_map = identify_combination_targets(basename(prefix), dirname(prefix); type = "iter", iterrun = iterrun)
    iterations = sort(collect(keys(iter_rank_map)))

    combined = Dict{String, Any}()

    for iteration in iterations
        filename = prefix * "$(iterrun)$(iteration).jld2"
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
    #         rm(prefix * "_run$(run).jld2")
    #     end
    # end
end

end