using MPI
using CUDA
using Adapt

MPI.Init()
atexit(MPI.Finalize)  

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

arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
grid = RectilinearGrid(arch;
    size = (12, 40, 10),
    halo = (1, 2, 1),              # <= This must be ≤ size
    x = (0, 1), y = (0, 1), z = (0, 1),
    topology = (Periodic, Periodic, Bounded)
)
free_surface = SplitExplicitFreeSurface(grid; substeps=1)
model = HydrostaticFreeSurfaceModel(; grid, free_surface, momentum_advection = VectorInvariant())
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")

data = jldopen(output_path * "test_Oceananigans_rank$(arch.local_rank).jld2")
u_data = data["u"]
v_data = data["v"]
w_data = data["w"]
@show time = data["clock"].time
@show iteration = data["clock"].iteration
close(data)

simulation = Simulation(model; Δt=0.01, stop_iteration=iteration+100)

simulation.model.clock.iteration = iteration
simulation.model.clock.time = time

@show maximum(u_data), minimum(u_data)
@show maximum(v_data), minimum(v_data)
@show maximum(w_data), minimum(w_data)
@show time, iteration

u_gpu = CuArray(u_data)
v_gpu = CuArray(v_data)
w_gpu = CuArray(w_data)

set!(model, u = u_gpu, v = v_gpu, w = w_gpu)

@show maximum(interior(model.velocities.u)), minimum(interior(model.velocities.u))
@show maximum(interior(model.velocities.v)), minimum(interior(model.velocities.v))
@show maximum(interior(model.velocities.w)), minimum(interior(model.velocities.w))

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = IterationInterval(20)

function progress(sim)
    u, v, w = sim.model.velocities
    # T, S, e = sim.model.ocean.model.tracers
    # Trange = (maximum((T)), minimum((T)))
    # Srange = (maximum((S)), minimum((S)))
    # erange = (maximum((e)), minimum((e)))

    umax = (maximum(abs, (u)),
            maximum(abs, (v)),
            maximum(abs, (w)))
        
    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    # msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Trange...)
    # msg4 = @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Srange...)
    # msg5 = @sprintf("extrema(e): (%.2f, %.2f) J, ", erange...)
    msg6 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg6

    wall_time[] = time_ns()

    jldsave(output_path * "test_rank$(arch.local_rank).jld2", 
    u=on_architecture(CPU(), interior(model.velocities.u)),
    v=on_architecture(CPU(), interior(model.velocities.v)),
    w=on_architecture(CPU(), interior(model.velocities.w)),
    clock = sim.model.clock)

    return nothing
end

add_callback!(simulation, progress, callback_interval)

# simulation.Δt = 5minutes 
# simulation.stop_time = target_time # 1 year 

run!(simulation)