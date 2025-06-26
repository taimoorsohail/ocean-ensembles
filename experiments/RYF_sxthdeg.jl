using MPI
using CUDA

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
using Oceananigans.Operators: Ax, Ay, Az, Δz
using Oceananigans.Fields: ReducedField
using ClimaOcean.EN4
using ClimaOcean.EN4: download_dataset
using ClimaOcean.DataWrangling.ETOPO
using Glob 

# File paths
data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

target_time = 20*365days

## Argument is provided by the submission script!

if isempty(ARGS)
    println("No arguments provided. Please enter architecture (CPU/GPU):")
    arch_input = readline()
    if arch_input == "GPU"
        arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
    elseif arch_input == "CPU"
        arch = CPU()
    else
        throw(ArgumentError("Invalid architecture. Must be 'CPU' or 'GPU'."))
    end
elseif ARGS[2] == "GPU"
    arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
elseif ARGS[2] == "CPU"
    arch = CPU()
else
    throw(ArgumentError("Architecture must be provided in the format julia --project example_script.jl --arch GPU"))
end    

total_ranks = MPI.Comm_size(MPI.COMM_WORLD)

@info "Using architecture: " * string(arch)

restartfiles = glob("checkpoint_iteration*", output_path)

# Extract the numeric suffix from each filename
restart_numbers = map(f -> parse(Int, match(r"checkpoint_iteration(\d+)", basename(f)).captures[1]), restartfiles)

iteration = 0
time = 0.0
if !isempty(restart_numbers) && maximum(restart_numbers) != 0
    # Extract the numeric suffix from each filename

    # Get the file with the maximum number
    clock_vars = jldopen(output_path * "checkpoint_iteration" * string(maximum(restart_numbers)) * "_rank" * string(arch.local_rank) * ".jld2")

    iteration = deepcopy(clock_vars["clock"].iteration)
    time = deepcopy(clock_vars["clock"].time)
    @info "Moving simulation to " * string(iteration) * " iterations"
    @info "Moving simulation to " * string(prettytime(time))

    close(clock_vars)
end

if time == target_time
    error("Terminating simulation at target time.")
end

# ### Download necessary files to run the code

# ### EN4 files
@info "Downloading/checking input data"

dates = vcat(collect(DateTime(1991, 1, 1): Month(1): DateTime(1991, 5, 1)),collect(DateTime(1990, 5, 1): Month(1): DateTime(1990, 12, 1)))

@info "We download the 1990-1991 data for an RYF implementation"

dataset = EN4Monthly() # Other options include ECCO2Monthly(), ECCO4Monthly() or ECCO2Daily()

temperature = Metadata(:temperature; dates, dataset = dataset, dir=data_path)
salinity    = Metadata(:salinity;    dates, dataset = dataset, dir=data_path)

download_dataset(temperature)
download_dataset(salinity)

# ### Grid and Bathymetry
@info "Defining grid"

Nx = Integer(360*6)
Ny = Integer(180*6)
Nz = Integer(75)

@info "Defining vertical z faces"

r_faces = exponential_z_faces(; Nz, depth=5000, h=12.43)
# z_faces = Oceananigans.MutableVerticalDiscretization(r_faces)

@info "Defining tripolar grid"

underlying_grid = TripolarGrid(arch;
                               size = (Nx, Ny, Nz),
                               z = r_faces,
                               halo = (7, 7, 4),
                               first_pole_longitude = 70,
                               north_poles_latitude = 55)

@info "Defining bottom bathymetry"

ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir = data_path)
ClimaOcean.DataWrangling.download_dataset(ETOPOmetadata)

@time bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
                                  minimum_depth = 10,
                                  interpolation_passes = 75, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
				                  major_basins = 2)

# For this bathymetry at this horizontal resolution we need to manually open the Gibraltar strait.
# view(bottom_height, 102:103, 124, 1) .= -400

@info "Defining grid"

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

# ### Restoring
#
# We include temperature and salinity surface restoring to ECCO data.

@info "Defining restoring rate"

restoring_rate  = 1 / 10days
z_below_surface = r_faces[end-1]

mask = LinearlyTaperedPolarMask(southern=(-80, -70), northern=(70, 90), z=(z_below_surface, 0))

FT = DatasetRestoring(temperature, grid; mask, rate=restoring_rate)
FS = DatasetRestoring(salinity,    grid; mask, rate=restoring_rate)
forcing = (T=FT, S=FS)

# ### Closures
# We include a Gent-McWilliam isopycnal diffusivity as a parameterization for the mesoscale
# eddy fluxes. For vertical mixing at the upper-ocean boundary layer we include the CATKE
# parameterization. We also include some explicit horizontal diffusivity.

@info "Defining closures"

using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation

momentum_advection = WENOVectorInvariant()
tracer_advection   = WENO(order=7)

free_surface = SplitExplicitFreeSurface(grid; substeps=70)

mixing_length = CATKEMixingLength(Cᵇ=0.01)
turbulent_kinetic_energy_equation = CATKEEquation(Cᵂϵ=1.0)

catke_closure = CATKEVerticalDiffusivity(ExplicitTimeDiscretization(); mixing_length, turbulent_kinetic_energy_equation) 
closure = (catke_closure, VerticalScalarDiffusivity(κ=1e-5, ν=1e-5))

@info "Defining ocean simulation"

@time ocean = ocean_simulation(grid;
                         momentum_advection,
                         tracer_advection,
                         free_surface,
                         closure,
                         forcing)

# ### Initial condition

# We initialize the ocean from the EN4 state estimate.

@info "Initialising with EN4"

set!(ocean.model, T=Metadata(:temperature; dates=first(dates), dataset = dataset, dir=data_path),
                  S=Metadata(:salinity;    dates=first(dates), dataset = dataset, dir=data_path))

# ### Atmospheric forcing

# We force the simulation with an JRA55-do atmospheric reanalysis.
@info "Defining Atmospheric state"

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=InMemory())

# ### Coupled simulation

# Now we are ready to build the coupled ocean--sea ice model and bring everything
# together into a `simulation`.

# We use a relatively short time step initially and only run for a few days to
# avoid numerical instabilities from the initial "shock" of the adjustment of the
# flow fields.

@info "Defining coupled model"

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=20, stop_time=60days)

simulation.model.ocean.model.clock.iteration = iteration
simulation.model.ocean.model.clock.time = time
simulation.model.atmosphere.clock.iteration = iteration
simulation.model.atmosphere.clock.time = time
simulation.model.clock.iteration = iteration
simulation.model.clock.time = time

# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = TimeInterval(5days)

function progress(sim)
    u, v, w = sim.model.ocean.model.velocities
    T, S, e = sim.model.ocean.model.tracers
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

tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)

#### TRACERS ####

volmask = CenterField(grid)
set!(volmask, 1)
wmask = ZFaceField(grid)

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
    @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1), condition = masks[j], suffix = suffixes[j]*"zonal")
    @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1,2), condition = masks[j], suffix = suffixes[j]*"depth")
    @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1,2,3), condition = masks[j], suffix = suffixes[j]*"tot")
end

@info "Merging velocity tuples"

transport_tuple = NamedTuple{Tuple(transport_names)}(Tuple(transport_outputs))

output_intervals = AveragedTimeInterval(5days)
checkpoint_intervals = TimeInterval(73days)

@info "Defining output writers"

simulation.output_writers[:ocean_tracer_content] = JLD2Writer(ocean.model, tracer_tuple;
                                                          dir = output_path,
                                                          schedule = output_intervals,
                                                          filename = "ocean_tracer_content_sxthdeg_iteration" * string(iteration),
                                                          overwrite_existing = false)

simulation.output_writers[:transport] = JLD2Writer(ocean.model, transport_tuple;
                                                          dir = output_path,
                                                          schedule = output_intervals,
                                                          filename = "mass_transport_sxthdeg_iteration" * string(iteration),
                                                          overwrite_existing = false)

simulation.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                                 dir = output_path,
                                                 schedule = output_intervals,
                                                 filename = "global_surface_fields_sxthdeg_iteration" * string(iteration),
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = false,
                                                 array_type = Array{Float32})

fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

simulation.output_writers[:fluxes] = JLD2Writer(ocean.model, fluxes;
                                                dir = output_path,
                                                schedule = output_intervals,
                                                filename = "fluxes_sxthdeg_iteration" * string(iteration),
                                                overwrite_existing = false)

@info "Saving restart"

function save_restart(sim)
    @info @sprintf("Saving checkpoint file")

    jldsave(output_path * "checkpoint_iteration" * string(sim.model.clock.iteration) * "_rank" * string(arch.local_rank) * ".jld2";
    u = on_architecture(CPU(), interior(sim.model.ocean.model.velocities.u)),
    v = on_architecture(CPU(), interior(sim.model.ocean.model.velocities.v)),
    w = on_architecture(CPU(), interior(sim.model.ocean.model.velocities.w)),
    T = on_architecture(CPU(), interior(sim.model.ocean.model.tracers.T)),
    S = on_architecture(CPU(), interior(sim.model.ocean.model.tracers.S)),
    e = on_architecture(CPU(), interior(sim.model.ocean.model.tracers.e)),
    clock = sim.model.clock)
end

add_callback!(simulation, save_restart, checkpoint_intervals)

if !isempty(restart_numbers) && maximum(restart_numbers) != 0
    @info "Loading with restart file from iteration " * string(maximum(restart_numbers))
    @info "Local rank ", arch.local_rank
    @info output_path * "checkpoint_iteration" * string(maximum(restart_numbers)) * "_rank" * string(arch.local_rank) * ".jld2"

    fields = jldopen(output_path * "checkpoint_iteration" * string(maximum(restart_numbers)) * "_rank" * string(arch.local_rank) * ".jld2")

    T_field = fields["T"]
    S_field = fields["S"]
    e_field = fields["e"]
    u_field = fields["u"]
    v_field = fields["v"]
    w_field = fields["w"]

    close(fields)

    set!(ocean.model, 
    T = (T_field),
    S = (S_field),
    u = (u_field),
    v = (v_field),
    w = (w_field),
    e = (e_field))

    u, v, w = ocean.model.velocities
    T, S, e = ocean.model.tracers

    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    erange = (maximum((e)), minimum((e)))

    umax = (maximum(abs, (u)),
            maximum(abs, (v)),
            maximum(abs, (w)))

    @info Trange, Srange, erange, umax
    
    @info "Restart found at " * string(prettytime(time))

    simulation.Δt = 5minutes 
    simulation.stop_time = target_time # 20 years 

    run!(simulation)
else
    @info "Running simulation"

    run!(simulation)

    simulation.Δt = 5minutes 
    simulation.stop_time = target_time # 20 years 

    run!(simulation)
end