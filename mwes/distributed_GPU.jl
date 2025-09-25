using MPI
using CUDA

MPI.Init()
atexit(MPI.Finalize)  

using Oceananigans
using Oceananigans.Units
using ClimaOcean
using Oceananigans.DistributedComputations
using Printf

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")

arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)

Nx, Ny, Nz = 100, 100, 50
Lx, Ly = 100, 100
@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialCoordinate(Nz, depth, 0) # <----- This doesn't work w/ mpirun -n 2 julia --project distributed_GPU.jl
# z_faces = (-6000,0) # <----- This also doesn't work w/ mpirun -n 2 julia --project distributed_GPU.jl

# z_faces = (-100,0) # <----- This works w/ mpirun -n 2 julia --project distributed_GPU.jl

@info "Creating grid"

grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (6, 6, 3))

@info "Creating model"
free_surface = SplitExplicitFreeSurface(grid; substeps = 70)

ocean_model = HydrostaticFreeSurfaceModel(; grid, free_surface)#, free_surface = ClimaOcean.OceanSimulations.default_free_surface(grid))
#                           closure = ClimaOcean.OceanSimulations.default_ocean_closure(),
#                           tracers = (:T, :S),
#                           free_surface = ClimaOcean.OceanSimulations.default_free_surface(grid),
#                           reference_density = 1020,
#                         #   rotation_rate = Ω_Earth,
#                         #   gravitational_acceleration = g_Earth,
#                         #   bottom_drag_coefficient = Default(0.003),
#                           biogeochemistry = nothing,
#                           timestepper = :SplitRungeKutta3,
#                         #   coriolis = Default(HydrostaticSphericalCoriolis(; rotation_rate)),
#                           momentum_advection = WENOVectorInvariant(),
#                           tracer_advection = WENO(order=7),
#                         #   equation_of_state = TEOS10EquationOfState(; reference_density),
#                           radiative_forcing = ClimaOcean.OceanSimulations.default_radiative_forcing(grid),
#                           warn = true,
#                           verbose = false)

# @time ocean = ocean_simulation(grid; free_surface)
@info "Creating simulation"

ocean = Simulation(ocean_model; Δt=10, verbose=false, stop_time=2hours)

# @info "Creating simulation"

# simulation = Simulation(ocean, Δt=10, stop_time=2hours)

# wizard = TimeStepWizard(cfl=1, max_change=1.1, max_Δt=1minute)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# # Nice progress messaging is helpful:

# ## Print a progress message
# progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
#                                 iteration(sim), prettytime(sim), prettytime(sim.Δt),
#                                 maximum(abs, sim.model.velocities.w), prettytime(sim.run_wall_time))

# add_callback!(simulation, progress_message, IterationInterval(40))
# @info "Running simulation"
# run!(simulation)

# @info "Simulation completed in " * prettytime(simulation.run_wall_time)