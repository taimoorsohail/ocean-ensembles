# using MPI
using CUDA

# MPI.Init()
# atexit(MPI.Finalize)  

using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
using Oceananigans.DistributedComputations
using Oceananigans.Operators: Ax, Ay, Az, Δz
using Oceananigans.Fields: ReducedField
using ClimaOcean.EN4
using ClimaOcean.EN4: download_dataset
using ClimaOcean.DataWrangling.ETOPO
using OceanEnsembles
using JLD2
using Test
using Glob
using Oceananigans.Architectures: on_architecture

arch = CPU()#Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
Nx, Ny, Nz = 10,10,10#360*6, 180*6, 75
@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
r_faces = ExponentialCoordinate(Nz, depth, 0)
@info "Defining grid"

grid = LatitudeLongitudeGrid(arch,
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             longitude = (0, 360),
                             latitude = (-70, 70),
                             z = r_faces)

free_surface = SplitExplicitFreeSurface(grid; substeps=70)

@info "Defining ocean model"

@time ocean = ocean_simulation(grid;
                         free_surface)

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")

# We force the simulation with an JRA55-do atmospheric reanalysis.
@info "Defining Atmospheric state"

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(20))

@info "Defining coupled model"

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=0.01, stop_iteration=110)

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = IterationInterval(5)

function progress(sim)
    T, S, e = sim.model.ocean.model.tracers
    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    erange = (maximum((e)), minimum((e)))
    η = sim.model.ocean.model.free_surface.η

    ηrange = (maximum((η)), minimum((η)))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
    # msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Trange...)
    msg4 = @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Srange...)
    msg5 = @sprintf("extrema(e): (%.2f, %.2f) J, ", erange...)
    msg6 = @sprintf("wall time: %s \n", prettytime(step_time))
    msg7 = @sprintf("extrema(η): (%.2f, %.2f) m, ", ηrange...)

    @info msg1 * msg3 * msg4 * msg5 * msg6 * msg7

    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, callback_interval)

output_intervals = TimeInterval(0.1)
# checkpoint_intervals = TimeInterval(365days)

#### SURFACE

@info "Defining surface outputs"

tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)
#=
#### TRACERS ####

volmask = CenterField(grid)
set!(volmask, 1)
wmask = ZFaceField(grid)
vmask = YFaceField(grid)
umask = XFaceField(grid)

@info "Defining condition masks"

Atlantic_mask = basin_mask(grid, "atlantic", volmask);
IPac_mask = basin_mask(grid, "indo-pacific", volmask);
glob_mask = Atlantic_mask .|| IPac_mask;

tracer_volmask = [Ax, Δz, volmask]
masks_centers = [repeat(glob_mask, 1, 1, size(volmask)[3]),
         repeat(Atlantic_mask, 1, 1, size(volmask)[3]),
         repeat(IPac_mask, 1, 1, size(volmask)[3])]
masks_wfaces = [repeat(glob_mask, 1, 1, size(wmask)[3]),
         repeat(Atlantic_mask, 1, 1, size(wmask)[3]),
         repeat(IPac_mask, 1, 1, size(wmask)[3])]

masks = [
            [masks_centers[1], masks_wfaces[1]],  # Global
            [masks_centers[2], masks_wfaces[2]],  # Atlantic
            [masks_centers[3], masks_wfaces[3]]   # IPac
        ]
        
@info "Tracers"

suffixes = ["_global_", "_atl_", "_ipac_"]
tracer_names = Symbol[]
tracer_outputs = Reduction[]

for j in 1:3
    @show j
    @time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[1], dims = (1), condition = masks[j][1], suffix = suffixes[j]*"zonal");
    @time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[2], dims = (1, 2), condition = masks[j][1], suffix = suffixes[j]*"depth");
    @time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[3], dims = (1, 2, 3), condition = masks[j][1], suffix = suffixes[j]*"tot");
end

@info "Merging tracer tuples"

tracer_tuple = NamedTuple{Tuple(tracer_names)}(Tuple(tracer_outputs))

#### VELOCITIES ####
@info "Velocities"

transport_volmask_operators = [Ax, Ay, Az]
transport_names = Symbol[]
transport_outputs = ReducedField[]
for j in 1:3
    @show j
    @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1), condition = masks[j], suffix = suffixes[j]*"zonal")
    @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1,2), condition = masks[j], suffix = suffixes[j]*"depth")
    @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1,2,3), condition = masks[j], suffix = suffixes[j]*"tot")
end

@info "Merging velocity tuples"

transport_tuple = NamedTuple{Tuple(transport_names)}(Tuple(transport_outputs))

@show transport_tuple
@show tracer_tuple

@info "Defining output writers"

@time simulation.output_writers[:ocean_tracer_content] = JLD2Writer(ocean.model, tracer_tuple;
                                                          dir = output_path,
                                                          schedule = output_intervals,
                                                          filename = "ocean_tracer_content_test_iteration" * string(Oceananigans.iteration(simulation)),
                                                          overwrite_existing = true)

@time simulation.output_writers[:transport] = JLD2Writer(ocean.model, transport_tuple;
                                                          dir = output_path,
                                                          schedule = output_intervals,
                                                          filename = "mass_transport_test_iteration" * string(Oceananigans.iteration(simulation)),
                                                          overwrite_existing = true)
=#
@time simulation.output_writers[:surfacetracers] = JLD2Writer(ocean.model, tracers;
                                                 dir = output_path,
                                                 schedule = TimeInterval(0.1),
                                                 filename = "global_surface_fields_test_tracers_iteration" * string(Oceananigans.iteration(simulation)),
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

@time simulation.output_writers[:surfacevels] = JLD2Writer(ocean.model, velocities;
                                                 dir = output_path,
                                                 schedule = TimeInterval(0.1),
                                                 filename = "global_surface_fields_test_velocities_iteration" * string(Oceananigans.iteration(simulation)),
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

run!(simulation)