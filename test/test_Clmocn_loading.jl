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

Nx, Ny, Nz = 10,10,10
depth = 6000meters
z_faces = exponential_z_faces(; Nz, depth)

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             z = z_faces,
                             latitude  = (-75, 75),
                             longitude = (0, 360))

free_surface = SplitExplicitFreeSurface(grid; substeps=1)
@time ocean = ocean_simulation(grid;
                         free_surface)

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")

# We force the simulation with an JRA55-do atmospheric reanalysis.
@info "Defining Atmospheric state"

radiation  = Radiation(arch)
@info "Loading atmosphere in memory"
atmosphere = JRA55PrescribedAtmosphere(arch; backend=InMemory())

@info "Defining coupled model"

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=5minutes, stop_time=20days)

# output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")

# data = jldopen(output_path * "test_ClimaOcean_rank$(arch.local_rank).jld2")
# T_data = data["T"]
# S_data = data["S"]
# e_data = data["e"]
# @show time = data["clock"].time
# @show iteration = data["clock"].iteration
# close(data)

time = 599529600.0
iteration = 28880

simulation.model.ocean.model.clock.iteration = iteration
simulation.model.ocean.model.clock.time = time
simulation.model.atmosphere.clock.iteration = iteration
simulation.model.atmosphere.clock.time = time
simulation.model.clock.iteration = iteration
simulation.model.clock.time = time
#=
@show maximum(T_data), minimum(T_data)
@show maximum(S_data), minimum(S_data)
@show maximum(e_data), minimum(e_data)
@show time, iteration

T_gpu = (T_data)
S_gpu = (S_data)
e_gpu = (e_data)

set!(ocean.model, T = T_gpu, S = S_gpu, e = e_gpu)

@show maximum(interior(ocean.model.tracers.T)), minimum(interior(ocean.model.tracers.T))
@show maximum(interior(ocean.model.tracers.S)), minimum(interior(ocean.model.tracers.S))
@show maximum(interior(ocean.model.tracers.e)), minimum(interior(ocean.model.tracers.e))

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = IterationInterval(20)

function progress(sim)
    T, S, e = sim.model.ocean.model.tracers
    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    erange = (maximum((e)), minimum((e)))

    # umax = (maximum(abs, (u)),
    #         maximum(abs, (v)),
    #         maximum(abs, (w)))
        
    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
    # msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Trange...)
    msg4 = @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Srange...)
    msg5 = @sprintf("extrema(e): (%.2f, %.2f) J, ", erange...)
    msg6 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg3 * msg4 * msg5 * msg6

    wall_time[] = time_ns()

    jldsave(output_path * "test_ClimaOcean_rank$(arch.local_rank).jld2", 
    T=on_architecture(CPU(), interior(sim.model.ocean.model.tracers.T)),
    S=on_architecture(CPU(), interior(sim.model.ocean.model.tracers.S)),
    e=on_architecture(CPU(), interior(sim.model.ocean.model.tracers.e)),
    clock = sim.model.clock)

    return nothing
end

add_callback!(simulation, progress, callback_interval)
=#
# simulation.Δt = 5minutes 
# simulation.stop_time = target_time # 1 year 

run!(simulation)