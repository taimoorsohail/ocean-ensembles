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

arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)

Nx, Ny, Nz = 100, 100, 50
Lx, Ly = 100, 100
@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialCoordinate(Nz, depth, 0) # <----- This doesn't work w/ mpirun -n 4 julia --project distributed_GPU.jl

grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), 
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces)

model = NonhydrostaticModel(; grid,
                            tracers = (:T, :S),
                            coriolis = FPlane(f=1e-4),
                            closure = AnisotropicMinimumDissipation())


simulation = Simulation(model, Δt=10, stop_time=2hours)

wizard = TimeStepWizard(cfl=1, max_change=1.1, max_Δt=1minute)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Nice progress messaging is helpful:

## Print a progress message
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.w), prettytime(sim.run_wall_time))

add_callback!(simulation, progress_message, IterationInterval(40))

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)
