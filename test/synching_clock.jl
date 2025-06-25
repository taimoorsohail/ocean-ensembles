using ClimaOcean
using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations

arch = GPU()

# using MPI
# using CUDA

# MPI.Init()
# atexit(MPI.Finalize)  

# arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)

Nx, Ny, Nz = 10,10,10
depth = 6000meters
z_faces = (0,depth)

# grid = LatitudeLongitudeGrid(arch;
#                              size = (Nx, Ny, Nz),
#                              halo = (7, 7, 7),
#                              z = z_faces,
#                              latitude  = (-75, 75),
#                              longitude = (0, 360))

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz),
                       halo = (1, 2, 1),              # <= This must be ≤ size
                       x = (0, 1), y = (0, 1), z = (0, 1),
                       topology = (Periodic, Periodic, Bounded))
                         
free_surface = SplitExplicitFreeSurface(grid; substeps=1)
@info "Defining Ocean"

# @time ocean = ocean_simulation(grid;
#                          free_surface)
@time ocean = HydrostaticFreeSurfaceModel(; grid,
                         free_surface)

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")

# We force the simulation with an JRA55-do atmospheric reanalysis.
# @info "Defining Atmosphere"

# radiation  = Radiation(arch)
# atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(20))

# @info "Defining coupled model"

# coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
# simulation = Simulation(coupled_model; Δt=5minutes, stop_time=20days)
simulation = Simulation(ocean; Δt=5minutes, stop_time=20days)

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")

time = 864000.0
iteration = 2880

ocean.clock.iteration = iteration
ocean.clock.time = time

# simulation.model.ocean.model.clock.iteration = iteration
# simulation.model.ocean.model.clock.time = time
# simulation.model.atmosphere.clock.iteration = iteration
# simulation.model.atmosphere.clock.time = time
# simulation.model.clock.iteration = iteration
# simulation.model.clock.time = time

# for field in simulation.model.ocean.model.timestepper.G⁻
#     fill!(field, 0)
# end

# for field in simulation.model.ocean.model.timestepper.Gⁿ
#     fill!(field, 0)
# end

run!(simulation)
