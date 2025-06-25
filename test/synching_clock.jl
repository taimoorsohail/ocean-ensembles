using ClimaOcean
using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Printf

arch = GPU()

Nx, Ny, Nz = 10,10,10
depth = 6000meters
z_faces = (0,depth)

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             z = z_faces,
                             latitude  = (-75, 75),
                             longitude = (0, 360))
                         
free_surface = SplitExplicitFreeSurface(grid; substeps=1)
@info "Defining Ocean"

@time ocean = ocean_simulation(grid;
                         free_surface)

@info "Defining Atmosphere"

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(20))

@info "Defining coupled model"

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=5minutes, stop_time=20days)

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")

time = 864000.0
iteration = 2880

@show simulation.model.ocean.model.clock.iteration = iteration
@show simulation.model.ocean.model.clock.time = time
@show simulation.model.atmosphere.clock.iteration = iteration
@show simulation.model.atmosphere.clock.time = time
@show simulation.model.clock.iteration = iteration
@show simulation.model.clock.time = 0.0

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = IterationInterval(20)

function progress(sim)
    T, S, e = sim.model.ocean.model.tracers

    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    erange = (maximum((e)), minimum((e)))
        
    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
    # msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Trange...)
    msg4 = @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Srange...)
    msg5 = @sprintf("extrema(e): (%.2f, %.2f) J, ", erange...)
    msg6 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg3 * msg4 * msg5 * msg6

    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, callback_interval)

# for field in simulation.model.ocean.model.timestepper.G⁻
#     fill!(field, 0)
# end

# for field in simulation.model.ocean.model.timestepper.Gⁿ
#     fill!(field, 0)
# end

run!(simulation)
