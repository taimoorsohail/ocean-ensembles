# using MPI
# using CUDA

# MPI.Init()
# atexit(MPI.Finalize)  

using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
using Oceananigans.DistributedComputations
using OceanEnsembles
using JLD2
using Test
using Glob
using Oceananigans.Architectures: on_architecture

arch = CPU()#Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
grid = RectilinearGrid(arch;
    size = (12, 40, 10),
    halo = (1, 2, 1),              # <= This must be ≤ size
    x = (0, 1), y = (0, 1), z = (0, 1),
    topology = (Periodic, Periodic, Bounded)
)
@info "Defining free surface"

free_surface = SplitExplicitFreeSurface(grid; substeps=1)
@info "Defining model"

model = HydrostaticFreeSurfaceModel(; grid, free_surface, tracers=(:T, :S, :e))
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
@info "Defining simulation"
simulation = Simulation(model; Δt=0.01, stop_iteration=110)

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = TimeInterval(2)

function progress(sim)
    u, v, w = sim.model.velocities
    T, S, e = sim.model.tracers
    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    erange = (maximum((e)), minimum((e)))

    umax = (maximum(abs, (u)),
            maximum(abs, (v)),
            maximum(abs, (w)))
        
    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Trange...)
    msg4 = @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Srange...)
    msg5 = @sprintf("extrema(e): (%.2f, %.2f) J, ", erange...)
    msg6 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg3 * msg4 * msg5 * msg6

    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, callback_interval)

#### SURFACE

@info "Defining surface outputs"

tracers = model.tracers
velocities = model.velocities

outputs = merge(tracers, velocities)

@info "Defining output writers"

output_intervals = TimeInterval(0.1)

@time simulation.output_writers[:surfacetracers] = JLD2Writer(model, tracers;
                                                 dir = output_path,
                                                 schedule = deepcopy(output_intervals),
                                                 filename = "global_surface_fields_test_tracersocngns",
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

@time simulation.output_writers[:surfacevels] = JLD2Writer(model, velocities;
                                                 dir = output_path,
                                                 schedule = deepcopy(output_intervals),
                                                 filename = "global_surface_fields_test_velocitiesocngns",
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

@info "Running simulation"

run!(simulation)

simulation.stop_iteration = 350

run!(simulation)