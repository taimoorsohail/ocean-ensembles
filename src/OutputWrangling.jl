module OutputWrangling

using Oceananigans
using Oceananigans.Fields: location
using JLD2
using ClimaOcean
using Glob
using Printf

export combine_ranks, identify_combination_targets

function grid_metrics(prefix, ranks)
    file   = jldopen(prefix * "_rank$(ranks[1]).jld2")
    data   = file["serialized/grid"]

    Nx, Ny, Nz = data.Nx, data.Ny*Integer(length(ranks)), data.Nz
    Hx, Hy, Hz = data.Hx, data.Hy, data.Hz
    nx = Integer(Nx / length(ranks))
    ny = Integer(Ny / length(ranks))

    depth = -6000.0 # Depth of the ocean in meters
    z_faces = ExponentialCoordinate(Nz, depth)
    return Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, z_faces
end

function create_grid(prefix, ranks; gridtype = "TripolarGrid")
    Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, z_faces = grid_metrics(prefix, ranks)
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
    Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, z_faces = grid_metrics(prefix, ranks)

    bottom_height = zeros(Nx, Ny)

    for rank in ranks
        irange = ny * rank + 1 : ny * (rank + 1)
        file   = jldopen(prefix * "_rank$(rank).jld2")
        data   = file["serialized/grid"].immersed_boundary.bottom_height[Hx+1:Nx+Hx, Hy+1:ny+Hy,  1]
        bottom_height[:, irange] .= data
        close(file)
    end

    return bottom_height
end

function combine_ranks(prefix, prefix_out; remove_split_files = false, gridtype = "TripolarGrid")
    iter_rank_map = identify_combination_targets(basename(prefix), dirname(prefix))
    iterations = collect(keys(iter_rank_map))
    for iteration in iterations
        @show iteration
        ranks = iter_rank_map[iteration]
        grid = create_grid(prefix * "_iteration$(iteration)", ranks; gridtype = gridtype)
        file0 = jldopen(prefix * "_iteration$(iteration)_rank$(ranks[1]).jld2")
        iters = keys(file0["timeseries/t"])
        @show iters, length(iters)
        times = Float64[file0["timeseries/t/$(iter)"] for iter in iters]
        close(file0)
        @info "Field accessed for iteration $(iteration) with ranks $(ranks)"

        utmp = FieldTimeSeries{Face,   Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration$(iteration).jld2", name="u")
        vtmp = FieldTimeSeries{Center, Face,   Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration$(iteration).jld2", name="v")
        wtmp = FieldTimeSeries{Center, Face,   Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration$(iteration).jld2", name="w")
        Ttmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration$(iteration).jld2", name="T")
        Stmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration$(iteration).jld2", name="S")
        etmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration$(iteration).jld2", name="e")
        @info "Defining distributed field time series for iteration $(iteration) with ranks $(ranks)"

        function set_distributed_field_time_series!(fts, prefix, ranks)
            Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, z_faces = grid_metrics(prefix * "_iteration$(iteration)", ranks)
            @info "grid_metrics: Nx=$(Nx), Ny=$(Ny), Nz=$(Nz), Hx=$(Hx), Hy=$(Hy), Hz=$(Hz), nx=$(nx), ny=$(ny)"
            field = Field{location(fts)...}(grid)
            Ny = size(fts, 2)

            # Move around this order so you don't open and close every rank/timestep
            for rank in ranks
                @show rank
                file   = jldopen(prefix * "_iteration$(iteration)_rank$(rank).jld2")
                irange = ny * rank + 1 : ny * (rank + 1)
                for (idx, iter) in enumerate(iters)
                    @show idx, iter
                    data   = file["timeseries/$(fts.name)/$(iter)"][:, :, 1]
                    interior(field, :, irange, 1) .= data
                    @time set!(fts, field, idx)
                end
                close(file)
            end
        end
        @info "Setting distributed field time series for iteration $(iteration) with ranks $(ranks)"

        set_distributed_field_time_series!(utmp, prefix_out, ranks)
        @show "Set u field time series for iteration $(iteration) with ranks $(ranks)"
        set_distributed_field_time_series!(vtmp, prefix_out, ranks)
        @show "Set v field time series for iteration $(iteration) with ranks $(ranks)"
        set_distributed_field_time_series!(wtmp, prefix_out, ranks)
        @show "Set w field time series for iteration $(iteration) with ranks $(ranks)"
        set_distributed_field_time_series!(Ttmp, prefix_out, ranks)
        @show "Set T field time series for iteration $(iteration) with ranks $(ranks)"
        set_distributed_field_time_series!(Stmp, prefix_out, ranks)
        @show "Set S field time series for iteration $(iteration) with ranks $(ranks)"
        set_distributed_field_time_series!(etmp, prefix_out, ranks)
        @show "Set e field time series for iteration $(iteration) with ranks $(ranks)"

        if remove_split_files
            for rank in ranks
                rm(prefix * "_iteration$(iteration)_rank$(rank).jld2")
            end
        end
        @show "Removed split files for iteration $(iteration) with ranks $(ranks)"
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