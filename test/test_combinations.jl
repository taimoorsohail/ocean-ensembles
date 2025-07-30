
using Oceananigans
using Oceananigans.Fields: location
using JLD2
using ClimaOcean
using Glob
using Printf
using MPI
using CUDA

MPI.Init()
atexit(MPI.Finalize)  

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")

if isempty(ARGS)
    println("No arguments provided. Please enter architecture (CPU/GPU):")
    arch_input = readline()
    if arch_input == "GPU"
        arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
    elseif arch_input == "CPU"
        arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
    else
        throw(ArgumentError("Invalid architecture. Must be 'CPU' or 'GPU'."))
    end
elseif ARGS[2] == "GPU"
    arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
elseif ARGS[2] == "CPU"
    arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
else
    throw(ArgumentError("Architecture must be provided in the format julia --project example_script.jl --arch GPU"))
end    

Nx = Integer(10)
Ny = Integer(10)
Nz = Integer(10)

grid = RectilinearGrid(size=(Nx,Ny,Nz), extent=(1,1,1), halo = (6, 6, 3))

free_surface = SplitExplicitFreeSurface(grid; substeps=70)

ocean = ocean_simulation(grid; free_surface)

tracers = ocean.model.tracers
velocities = ocean.model.velocities
outputs = merge(tracers, velocities)

output_intervals = TimeInterval(1days)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                                        dir = output_path,
                                                        schedule = output_intervals,
                                                        filename = "global_surface_fields_test_iteration" * string(Oceananigans.iteration(ocean)),
                                                        indices = (:, :, grid.Nz),
                                                        with_halos = false,
                                                        overwrite_existing = true,
                                                        array_type = Array{Float32})

















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
        underlying_grid = LatitudeLongitudeGrid(CPU();
                                     size = (Nx, Ny, Nz),
                                     z = z_faces,
                                     halo = (Hx, Hy, Hz),
                                     latitude  = (-75, 75),
                                     longitude = (0, 360))        
    elseif gridtype == "TripolarGrid"
        underlying_grid = TripolarGrid(CPU();
                            size = (Nx, Ny, Nz),
                            z = z_faces,
                            halo = (Hx, Hy, Hz),
                            first_pole_longitude = 70,
                            north_poles_latitude = 55)
    end

    bottom_height = read_bathymetry(prefix, ranks)

    grid  = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
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
    iterations = [0]
    
    for iteration in iterations
        @show iteration
        ranks = [0,1,2]
        grid = create_grid(prefix * "_iteration$(iteration)", ranks; gridtype = gridtype)
        file0 = jldopen(prefix * "_iteration$(iteration)_rank$(ranks[1]).jld2")
        iters = keys(file0["timeseries/t"])
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
            field = Field{location(fts)...}(grid)
            Ny = size(fts, 2)
            
            # Move around this order so you don't open and close every rank/timestep
            for rank in ranks
                @show rank
                file   = jldopen(prefix * "_iteration$(iteration)_rank$(rank).jld2")
                irange = ny * rank + 1 : ny * (rank + 1)
                for (idx, iter) in enumerate(iters)
                    # @show idx/length(iters)
                    data   = file["timeseries/$(fts.name)/$(iter)"][:, :, 1]
                    @show size(data), size(interior(fts[idx], :, irange, 1))
                    # interior(fts[idx], :, irange, 1) .= data
                end
                close(file)
            end
        end
        @info "Setting distributed field time series for iteration $(iteration) with ranks $(ranks)"
        @show utmp[1]
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

prefix = "/g/data/v46/txs156/ocean-ensembles/outputs/global_surface_fields_onedeg"
prefix_out = prefix
combine_ranks(prefix, prefix; remove_split_files = false, gridtype = "TripolarGrid")

# iters = [0]

# data = jldopen(prefix * "_iteration0_rank0.jld2")
# file = data["timeseries/u/839664"]
# Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, z_faces = grid_metrics(prefix * "_iteration0", [0,1,2])

# grid = create_grid(prefix * "_iteration0", [0,1,2]; gridtype = "LatitudeLongitudeGrid")
# times = Float64[data["timeseries/t/$(iter)"] for iter in iters]

# utmp = FieldTimeSeries{Face,   Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * "_iteration0.jld2", name="u")

# irange = ny * 0 + 1 : ny * (0 + 1)

# @time interior(utmp[1])[:, irange, 1] .= file[:, :, 1]

# @show maximum(utmp[1])