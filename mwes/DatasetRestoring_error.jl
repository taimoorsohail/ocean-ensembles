using MPI
using CUDA

MPI.Init()
atexit(MPI.Finalize)  

using Oceananigans
using Oceananigans.Units
using ClimaOcean
using Oceananigans.DistributedComputations
using Printf
using Dates
using ClimaOcean.EN4
using ClimaOcean.ECCO
using ClimaOcean.EN4: download_dataset

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")

arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)

function memory_status(arch::Union{Distributed{<:GPU}, <:GPU})
    free, total = CUDA.memory_info()
    used = total - free
    used_GiB, free_GiB, total_GiB = used / 2^30, free / 2^30, total / 2^30
    @show arch.local_rank, used_GiB, free_GiB, total_GiB
end

function memory_status(arch::Union{Distributed{<:CPU}, <:CPU})
    total = Sys.total_memory()
    free  = Sys.free_memory()
    used  = total - free
    used_GiB, free_GiB, total_GiB = used / 2^30, free / 2^30, total / 2^30
    @show arch.local_rank, used_GiB, free_GiB, total_GiB
end

Nx, Ny, Nz = 100, 100, 50
Lx, Ly = 100, 100
@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialCoordinate(Nz, depth, 0) 
memory_status(arch)
@info "Creating grid"

underlying_grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (6, 6, 3))

@info "Defining bottom bathymetry"

memory_status(arch)

ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir = data_path)
ClimaOcean.DataWrangling.download_dataset(ETOPOmetadata)
memory_status(arch)
@time bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
                                  minimum_depth = 15,
                                  interpolation_passes = 1, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
				                  major_basins = 2)

memory_status(arch)

@info "Defining grid"

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

memory_status(arch)
### Restoring

# We include surface salinity restoring to a predetermined dataset.

dates = vcat(collect(DateTime(1991, 1, 1): Month(1): DateTime(1991, 5, 1)),
             collect(DateTime(1990, 5, 1): Month(1): DateTime(1990, 12, 1)))

@info "We download the 1990-1991 data for an RYF implementation"

dataset = EN4Monthly() # Other options include ECCO2Monthly(), ECCO4Monthly() or ECCO2Daily()

temperature = Metadata(:temperature; dates, dataset = dataset, dir=data_path)
salinity    = Metadata(:salinity;    dates, dataset = dataset, dir=data_path)

download_dataset(temperature)
download_dataset(salinity)

memory_status(arch)

@info "Defining restoring rate"

restoring_rate  = 1 / 18days
@inline mask(x, y, z, t) = z ≥ z_surf - 1

FS = DatasetRestoring(salinity, grid; mask, rate=restoring_rate, time_indices_in_memory = 10)
forcing = (; S=FS)

memory_status(arch)

@info "Creating model"
free_surface = SplitExplicitFreeSurface(grid; substeps = 70)

memory_status(arch)
ocean_model = HydrostaticFreeSurfaceModel(; grid, free_surface, timestepper = :SplitRungeKutta3)
memory_status(arch)

@info "Creating simulation"

simulation = Simulation(ocean_model; Δt=10, verbose=false, stop_time=2hours)

memory_status(arch)

wizard = TimeStepWizard(cfl=1, max_change=1.1, max_Δt=1minutes)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Nice progress messaging is helpful:

## Print a progress message
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.w), prettytime(sim.run_wall_time))

add_callback!(simulation, progress_message, IterationInterval(40))
@info "Running simulation"
run!(simulation)

# @info "Simulation completed in " * prettytime(simulation.run_wall_time)