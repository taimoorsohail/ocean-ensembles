module OutputWrangling

using Oceananigans
using Oceananigans.Fields: location
using JLD2
using ClimaOcean
using Glob
using Printf

export combine_outputs, identify_combination_targets

function grid_metrics(prefix, ranks)
    file   = jldopen(prefix * "_rank$(ranks[1]).jld2")
    data   = file["serialized/grid"]

    Nx, Ny, Nz = data.Nx, data.Ny*Integer(length(ranks)), data.Nz
    Hx, Hy, Hz = data.Hx, data.Hy, data.Hz
    nx = Integer(Nx / length(ranks))
    ny = Integer(Ny / length(ranks))

    r_faces = exponential_z_faces(; Nz, depth=5000, h=34)
    z_faces = Oceananigans.MutableVerticalDiscretization(r_faces)
    return Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, z_faces
end

function create_grid(prefix, ranks)
    Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, z_faces = grid_metrics(prefix, ranks)
    @warn "We assume a tripolar grid when combining outputs! If this is not the case, please modify the source code..."
    grid = TripolarGrid(CPU();
                        size = (Nx, Ny, Nz),
                        z = z_faces,
                        halo = (Hx, Hy, Hz),
                        first_pole_longitude = 70,
                        north_poles_latitude = 55)

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

function combine_outputs(prefix, prefix_out; remove_split_files = false)
    iter_rank_map = identify_combination_targets(basename(prefix), dirname(prefix))
    iterations = collect(keys(iter_rank_map))
    for iteration in iterations
        ranks = iter_rank_map[iteration]
        grid = create_grid(prefix * "_iteration$(iteration)", ranks)
        file0 = jldopen(prefix * "_iteration$(iteration)_rank$(ranks[1]).jld2")
        iters = keys(file0["timeseries/t"])
        times = Float64[file0["timeseries/t/$(iter)"] for iter in iters]
        close(file0)

        utmp = FieldTimeSeries{Face,   Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration$(iteration).jld2", name="u")
        vtmp = FieldTimeSeries{Center, Face,   Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration$(iteration).jld2", name="v")
        wtmp = FieldTimeSeries{Center, Face,   Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration$(iteration).jld2", name="w")
        Ttmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration$(iteration).jld2", name="T")
        Stmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration$(iteration).jld2", name="S")
        etmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration$(iteration).jld2", name="e")

        function set_distributed_field_time_series!(fts, prefix, ranks)
            Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, z_faces = grid_metrics(prefix * "_iteration$(iteration)", ranks)

            field = Field{location(fts)...}(grid)
            Ny = size(fts, 2)
            for (idx, iter) in enumerate(iters)
                for rank in ranks
                    irange = ny * rank + 1 : ny * (rank + 1)
                    file   = jldopen(prefix * "_iteration$(iteration)_rank$(rank).jld2")
                    data   = file["timeseries/$(fts.name)/$(iter)"][:, :, 1]

                    interior(field, :, irange, 1) .= data
                    close(file)
                end

                set!(fts, field, idx)
            end
        end

        set_distributed_field_time_series!(utmp, prefix_out * "_iteration$(iteration)", ranks)
        set_distributed_field_time_series!(vtmp, prefix_out * "_iteration$(iteration)", ranks)
        set_distributed_field_time_series!(wtmp, prefix_out * "_iteration$(iteration)", ranks)
        set_distributed_field_time_series!(Ttmp, prefix_out * "_iteration$(iteration)", ranks)
        set_distributed_field_time_series!(Stmp, prefix_out * "_iteration$(iteration)", ranks)
        set_distributed_field_time_series!(etmp, prefix_out * "_iteration$(iteration)", ranks)

        if remove_split_files
            for rank in ranks
                rm(prefix * "_iteration$(iteration)_rank$(rank).jld2")
            end
        end
    end
    return nothing
end 

function identify_combination_targets(prefix, output_path)
    file_pattern = prefix * "_iteration*_rank*"
    files = glob(file_pattern, output_path)

    pattern = Regex("^" * prefix * "_iteration(\\d+)_rank(\\d+)\\.jld2")
    # Step 1: Parse iteration and rank from filenames
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
end
end