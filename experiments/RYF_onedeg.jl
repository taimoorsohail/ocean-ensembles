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
using Oceananigans.Architectures: on_architecture
using JLD2

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

target_time = 365days*25 # 25 years
checkpoint_type = "none" # "none", "last", "first"

## Argument is provided by the submission script!

if isempty(ARGS)
    println("No arguments provided. Please enter architecture (CPU/GPU):")
    arch_input = readline()
    if arch_input == "GPU"
        arch = GPU()
    elseif arch_input == "CPU"
        arch = CPU()
    else
        throw(ArgumentError("Invalid architecture. Must be 'CPU' or 'GPU'."))
    end
elseif ARGS[2] == "GPU"
    arch = GPU()
elseif ARGS[2] == "CPU"
    arch = CPU()
else
    throw(ArgumentError("Architecture must be provided in the format julia --project example_script.jl --arch GPU"))
end    

#total_ranks = MPI.Comm_size(MPI.COMM_WORLD)

@info "Using architecture: " * string(arch)

restartfiles = glob("checkpoint_onedeg_iteration*", output_path)

# Extract the numeric suffix from each filename
restart_numbers = map(f -> parse(Int, match(r"checkpoint_onedeg_iteration(\d+)", basename(f)).captures[1]), restartfiles)

if !isempty(restart_numbers) && maximum(restart_numbers) != 0 && checkpoint_type != "none"
    # Extract the numeric suffix from each filename

    # Get the file with the maximum number
    if checkpoint_type == "last"
        clock_vars = jldopen(output_path * "checkpoint_onedeg_iteration" * string(maximum(restart_numbers)) * ".jld2")
    elseif checkpoint_type == "first"
        clock_vars = jldopen(output_path * "checkpoint_onedeg_iteration" * string(minimum(restart_numbers)) * ".jld2")
    end

    iteration = deepcopy(clock_vars["clock"].iteration)
    time = deepcopy(clock_vars["clock"].time)
    last_Δt = deepcopy(clock_vars["clock"].last_Δt)   

    @info "Moving simulation to " * string(iteration) * " iterations"
    @info "Moving simulation to " * string(prettytime(time))
    @info "Moving simulation last_dt to " * string(last_Δt)

    close(clock_vars)
end

if time == target_time
    error("Terminating simulation at target time.")
end

# ### Download necessary files to run the code

# ### ECCO files
@info "Downloading/checking input data"

dates = vcat(collect(DateTime(1991, 1, 1): Month(1): DateTime(1991, 5, 1)),
             collect(DateTime(1990, 5, 1): Month(1): DateTime(1990, 12, 1)))

@info "We download the 1990-1991 data for an RYF implementation"

dataset = EN4Monthly() # Other options include ECCO2Monthly(), ECCO4Monthly() or ECCO2Daily()

temperature = Metadata(:temperature; dates, dataset = dataset, dir=data_path)
salinity    = Metadata(:salinity;    dates, dataset = dataset, dir=data_path)

download_dataset(temperature)
download_dataset(salinity)

# ### Grid and Bathymetry
@info "Defining grid"

Nx = Integer(360)
Ny = Integer(180)
Nz = Integer(75)

@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
r_faces = ExponentialCoordinate(Nz, depth, 0, scale = 0.25*-depth)

@info "Top grid cell is " * string(abs(round(r_faces(Nz)))) * "m thick"

r_faces = Oceananigans.Grids.MutableVerticalDiscretization(r_faces)

@info "Defining tripolar grid"

underlying_grid = TripolarGrid(arch;
                               size = (Nx, Ny, Nz),
                               z = r_faces,
                               halo = (5, 5, 4),
                               first_pole_longitude = 70,
                               north_poles_latitude = 55)

@info "Defining bottom bathymetry"

ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir = data_path)
ClimaOcean.DataWrangling.download_dataset(ETOPOmetadata)

@time bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
                                  minimum_depth = 15,
                                  interpolation_passes = 1, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
				                  major_basins = 2)
# view(bottom_height, 73:78, 88:89, 1) .= -1000 # open Gibraltar strait

@info "Defining grid"

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

# ### Restoring
#
# We include temperature and salinity restoring to a predetermined dataset.

# @info "Defining restoring rate"

restoring_rate  = 1 / 3days
mask = LinearlyTaperedPolarMask(southern=(-80, -70), northern=(70, 90), z=(-15, 0))

FT = DatasetRestoring(temperature, grid; mask, rate=restoring_rate)
FS = DatasetRestoring(salinity,    grid; mask, rate=restoring_rate)
forcing = (T=FT, S=FS)

# ### Closures
# We include a Gent-McWilliam isopycnal diffusivity as a parameterization for the mesoscale
# eddy fluxes. For vertical mixing at the upper-ocean boundary layer we include the CATKE
# parameterization. We also include some explicit horizontal diffusivity.

@info "Defining closures"

eddy_closure = Oceananigans.TurbulenceClosures.IsopycnalSkewSymmetricDiffusivity(κ_skew=2e3, κ_symmetric=2e3)
vertical_mixing = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity(minimum_tke=1e-6)
horizontal_viscosity = HorizontalScalarDiffusivity(ν=4000)
closure = (eddy_closure, horizontal_viscosity, vertical_mixing)

# ### Ocean simulation
# Now we bring everything together to construct the ocean simulation.
# We use a split-explicit timestepping with 30 substeps for the barotropic
# mode.

@info "Defining free surface"

free_surface       = SplitExplicitFreeSurface(grid; substeps=70)
momentum_advection = WENOVectorInvariant(order = 5)
tracer_advection   = WENO(order=5)

@info "Defining ocean simulation"

@time ocean = ocean_simulation(grid;
                         momentum_advection,
                         tracer_advection,
                         free_surface,
                         closure)

# ### Initial condition

# We initialize the ocean from the ECCO state estimate.

@info "Initialising with EN4"

set!(ocean.model, T=Metadata(:temperature; dates=first(dates), dataset = dataset, dir=data_path),
                  S=Metadata(:salinity;    dates=first(dates), dataset = dataset, dir=data_path))

# ### Atmospheric forcing

# We force the simulation with an JRA55-do atmospheric reanalysis.
@info "Defining Atmospheric state"

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(25), include_rivers_and_icebergs=true)

# ### Coupled simulation

# Now we are ready to build the coupled ocean--sea ice model and bring everything
# together into a `simulation`.

# We use a relatively short time step initially and only run for a few days to
# avoid numerical instabilities from the initial "shock" of the adjustment of the
# flow fields.

@info "Defining coupled model"
@time coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

simulation = Simulation(coupled_model; Δt=6minutes, stop_time=60days)

# ### Restarting the simulation
if !isempty(restart_numbers) && maximum(restart_numbers) != 0 && checkpoint_type != "none"
    simulation.model.ocean.model.clock.iteration = iteration
    simulation.model.ocean.model.clock.time = time
    simulation.model.atmosphere.clock.iteration = iteration
    simulation.model.atmosphere.clock.time = time
    simulation.model.clock.iteration = iteration
    simulation.model.clock.time = time
    time_step!(atmosphere, 0)
    simulation.model.atmosphere.clock.iteration -= 1
    simulation.model.ocean.model.clock.last_Δt = last_Δt
end

# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = TimeInterval(1days)

function progress(sim)
    η = sim.model.ocean.model.free_surface.η
    u, v, w = sim.model.ocean.model.velocities
    T, S, e = sim.model.ocean.model.tracers

    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    erange = (maximum((e)), minimum((e)))
    ηrange = (maximum((η)), minimum((η)))

    umax = (maximum(abs, (u)),
            maximum(abs, (v)),
            maximum(abs, (w)))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s,", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Trange...)
    msg4 = @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Srange...)
    msg5 = @sprintf("extrema(e): (%.2f, %.2f) J, ", erange...)
    msg6 = @sprintf("extrema(η): (%.2f, %.2f) m, ", ηrange...)
    msg7 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg3 * msg4 * msg5 * msg6 * msg7
    
    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, callback_interval)
checkpoint_intervals = TimeInterval(365days)

#### REGRIDDING ####

@info "Defining destination underlying grid"
underlying_destination_grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             z = r_faces,
                             latitude  = (-80, 90),
                             longitude = (0, 360))

@info "Interpolating bottom bathymetry"

@time bottom_height = regrid_bathymetry(underlying_destination_grid, ETOPOmetadata;
                                  minimum_depth = 15,
                                  interpolation_passes = 1, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
				                  major_basins = 2)
# view(bottom_height, 73:78, 88:89, 1) .= -1000 # open Gibraltar strait
                                  
@info "Defining destination grid"

@time destination_grid = ImmersedBoundaryGrid(underlying_destination_grid, GridFittedBottom(bottom_height); active_cells_map=true)

destination_field = Field{Center, Center, Center}(destination_grid)
source_field = Field{Center, Center, Center}(grid)

@info "Defining regridder weights"
W_bilin = regridder_weights!(source_field, destination_field; method = "bilinear")
W_cons = regridder_weights!(source_field, destination_field; method = "conservative")

@info "Defining outputs - bilinearly interpolated"

tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)
output_names_bilin = Symbol[]
outputs_bilin = Field[]
output_names_cons = Symbol[]
outputs_cons = Field[]

for key in keys(outputs)
    @info "Regridding output: " * string(key)
    f = outputs[key]
    f_dst_bilin = regrid_tracers!(f, destination_field, W_bilin)                       
    push!(outputs_bilin, f_dst_bilin)
    push!(output_names_bilin, Symbol(key, "_bilinear"))
    f_dst_cons = regrid_tracers!(f, destination_field, W_cons)                       
    push!(outputs_cons, f_dst_cons)
    push!(output_names_cons, Symbol(key, "_conservative"))
end

#### VERTICAL INTEGRALS ####
@info "Defining vertical integral outputs"

vertical_integral_conservative = Symbol[]
vertical_integral_conservative_outputs = Field[]

for key in keys(conservative_tuple)
    f = conservative_tuple[key]
    f_int = CumulativeIntegral(f, dims = 3)
    push!(vertical_integral_conservative_outputs, f_int)
    push!(vertical_integral_conservative, Symbol(key, "_cumintegral"))
end

bilinear_tuple = NamedTuple{Tuple(output_names_bilin)}(Tuple(outputs_bilin))
conservative_tuple = NamedTuple{Tuple(output_names_cons)}(Tuple(outputs_cons))
vertical_integral_tuple = NamedTuple{Tuple(vertical_integral_conservative)}(Tuple(vertical_integral_conservative_outputs))

#### OUTPUTS ####
iteration_number = string(Oceananigans.iteration(simulation))

@info "Defining slice outputs"

depths = [0,-100, -500, -1000, -2000]

symbols_slice = Symbol[]  # empty vector to store symbols
symbols_cumint = Symbol[]  # empty vector to store symbols

for (ind, depth) in enumerate(depths)
    pln, ind_pln =  findmin(abs.(grid.z.cᵃᵃᶜ[1:Nz] .- depths[ind]))
    push!(symbols_slice, Symbol("plane_$(abs(round(pln, digits=1)))m"))
    push!(symbols_cumint, Symbol("integrated_$(abs(round(pln, digits=1)))m"))

    @time simulation.output_writers[symbols_slice[ind]] = JLD2Writer(ocean.model, bilinear_tuple;
                                                dir = output_path,
                                                schedule = TimeInterval(1days),
                                                filename = "global_*" * string(round(pln)) * "m_fields_onedeg_iteration" * iteration_number,
                                                indices = (:, :, ind_pln),
                                                with_halos = false,
                                                overwrite_existing = true,
                                                array_type = Array{Float32})

    @time simulation.output_writers[symbols_cumint[ind]] = JLD2Writer(ocean.model, vertical_integral_tuple;
                                                dir = output_path,
                                                schedule = TimeInterval(1days),
                                                filename = "global_*" * string(round(pln)) * "m_integral_onedeg_iteration" * iteration_number,
                                                indices = (:, :, ind_pln),
                                                with_halos = false,
                                                overwrite_existing = true,
                                                array_type = Array{Float32})
end



#=
#### TRACERS ####

@info "Defining tracer outputs - conservatively regridded"

# volmask = CenterField(destination_grid)
# set!(volmask, 1)
# wmask = ZFaceField(destination_grid)

# @info "Defining condition masks"

# Atlantic_mask = basin_mask(destination_grid, "atlantic", volmask);
# IPac_mask = basin_mask(destination_grid, "indo-pacific", volmask);
# glob_mask = Atlantic_mask .|| IPac_mask;

# tracer_volmask = [Ax, Δz, volmask]
# masks_centers = [repeat(glob_mask, 1, 1, size(volmask)[3]),
#          repeat(Atlantic_mask, 1, 1, size(volmask)[3]),
#          repeat(IPac_mask, 1, 1, size(volmask)[3])]
# masks_wfaces = [repeat(glob_mask, 1, 1, size(wmask)[3]),
#          repeat(Atlantic_mask, 1, 1, size(wmask)[3]),
#          repeat(IPac_mask, 1, 1, size(wmask)[3])]

# masks = [
#             [masks_centers[1], masks_wfaces[1]],  # Global
#             [masks_centers[2], masks_wfaces[2]],  # Atlantic
#             [masks_centers[3], masks_wfaces[3]]   # IPac
#         ]

# suffixes = ["_global_", "_atl_", "_ipac_"]
tracer_names = Symbol[]
tracer_outputs = Reduction[]

@info "Tracers"
@time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, dims = (3));

tracer_tuple = NamedTuple{Tuple(tracer_names)}(Tuple(tracer_outputs))


# for j in 1:3
#     # @time ocean_tracer_content!(tracer_names, tracer_outputs; dst_field = destination_field, weights = W_cons, outputs=tracers, operator = tracer_volmask[2], dims = (1, 2), condition = masks[j][1], suffix = suffixes[j]*"depth");
#     # @time ocean_tracer_content!(tracer_names, tracer_outputs; dst_field = destination_field, weights = W_cons, outputs=tracers, operator = tracer_volmask[3], dims = (1, 2, 3), condition = masks[j][1], suffix = suffixes[j]*"tot");
# end

@info "Merging tracer tuples"


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

@info "Defining output writers"


@time simulation.output_writers[:transport] = JLD2Writer(ocean.model, transport_tuple;
                                                          dir = output_path,
                                                          schedule = TimeInterval(1days),
                                                          filename = "mass_transport_onedeg_iteration" * iteration_number,
                                                          overwrite_existing = true)



@time simulation.output_writers[:ocean_tracer_content] = JLD2Writer(ocean.model, tracer_tuple;
                                                          dir = output_path,
                                                          schedule = TimeInterval(1days),
                                                          filename = "ocean_tracer_content_onedeg_iteration" * iteration_number,
                                                          overwrite_existing = true)

=#
if checkpoint_type != "none"
    @info "Removing all checkpoints"
    for f in restartfiles
        if isfile(f)
            @info "Removing old restart file: $f"
            rm(f; force = true)
        end
    end
end

@info "Saving restart"

function save_restart(sim)
    @info @sprintf("Saving checkpoint file")

    jldsave(output_path * "checkpoint_onedeg_iteration" * string(sim.model.clock.iteration) * ".jld2";
    u = on_architecture(CPU(), interior(sim.model.ocean.model.velocities.u)),
    v = on_architecture(CPU(), interior(sim.model.ocean.model.velocities.v)),
    w = on_architecture(CPU(), interior(sim.model.ocean.model.velocities.w)),
    T = on_architecture(CPU(), interior(sim.model.ocean.model.tracers.T)),
    S = on_architecture(CPU(), interior(sim.model.ocean.model.tracers.S)),
    e = on_architecture(CPU(), interior(sim.model.ocean.model.tracers.e)),
    u_Gⁿ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.Gⁿ.u)),
    v_Gⁿ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.Gⁿ.v)),
    U_Gⁿ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.Gⁿ.U)),
    V_Gⁿ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.Gⁿ.V)),
    T_Gⁿ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.Gⁿ.T)),
    S_Gⁿ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.Gⁿ.S)),
    e_Gⁿ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.Gⁿ.e)),
    u_G⁻ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.G⁻.u)),
    v_G⁻ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.G⁻.v)),
    U_G⁻ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.G⁻.U)),
    V_G⁻ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.G⁻.V)),
    T_G⁻ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.G⁻.T)),
    S_G⁻ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.G⁻.S)),
    e_G⁻ = on_architecture(CPU(), interior(sim.model.ocean.model.timestepper.G⁻.e)),
    clock = sim.model.ocean.model.clock)

    restartfiles = glob("checkpoint_onedeg_iteration*", output_path)

    # Extract the numeric suffix from each filename
    restart_numbers = map(f -> parse(Int, match(r"checkpoint_onedeg_iteration(\d+)", basename(f)).captures[1]), restartfiles)

    sorted_restart_numbers = sort(unique(restart_numbers))

    # Keep only the last 3 iteration numbers
    if length(sorted_restart_numbers) < 3
        keep = sorted_restart_numbers
    else
        # Keep the last 3 iterations
        @info "Keeping last 3 restart files: " * string(sorted_restart_numbers[end-2:end])
        @info "Removing older restart files"
        keep = sorted_restart_numbers[end-2:end]
    end
    
    # Loop through and remove all older files for this rank
    for number in sorted_restart_numbers
        if number ∉ keep
            filename = output_path * "checkpoint_onedeg_iteration$(number).jld2"
            if isfile(filename)
                @info "Removing old restart file: $filename"
                rm(filename; force = true)
            end
        end
    end
end

add_callback!(simulation, save_restart, checkpoint_intervals)

if !isempty(restart_numbers) && maximum(restart_numbers) != 0 && checkpoint_type != "none"
    if checkpoint_type == "last"
        @info "Restarting from last checkpoint at iteration " * string(maximum(restart_numbers))
        fields_loaded = jldopen(output_path * "checkpoint_onedeg_iteration" * string(maximum(restart_numbers)) * ".jld2")
    elseif checkpoint_type == "first"
        @info "Restarting from first checkpoint at iteration " * string(minimum(restart_numbers))
        fields_loaded = jldopen(output_path * "checkpoint_onedeg_iteration" * string(minimum(restart_numbers)) * ".jld2")
    end

    T_field = fields_loaded["T"]
    S_field = fields_loaded["S"]
    e_field = fields_loaded["e"]
    u_field = fields_loaded["u"]
    v_field = fields_loaded["v"]
    w_field = fields_loaded["w"]

    T_field_Gⁿ = fields_loaded["T_Gⁿ"]
    S_field_Gⁿ = fields_loaded["S_Gⁿ"]
    e_field_Gⁿ = fields_loaded["e_Gⁿ"]
    u_field_Gⁿ = fields_loaded["u_Gⁿ"]
    v_field_Gⁿ = fields_loaded["v_Gⁿ"]
    U_field_Gⁿ = fields_loaded["U_Gⁿ"]
    V_field_Gⁿ = fields_loaded["V_Gⁿ"]

    T_field_G⁻ = fields_loaded["T_G⁻"]
    S_field_G⁻ = fields_loaded["S_G⁻"]
    e_field_G⁻ = fields_loaded["e_G⁻"]
    u_field_G⁻ = fields_loaded["u_G⁻"]
    v_field_G⁻ = fields_loaded["v_G⁻"]
    U_field_G⁻ = fields_loaded["U_G⁻"]
    V_field_G⁻ = fields_loaded["V_G⁻"]

    close(fields_loaded)

    set!(ocean.model, 
    T = (T_field),
    S = (S_field),
    u = (u_field),
    v = (v_field),
    w = (w_field),
    e = (e_field))

    set!(ocean.model.timestepper.Gⁿ, 
    T = (T_field_Gⁿ),
    S = (S_field_Gⁿ),
    u = (u_field_Gⁿ),
    v = (v_field_Gⁿ),
    U = (U_field_Gⁿ),
    V = (V_field_Gⁿ),
    e = (e_field_Gⁿ))

    set!(ocean.model.timestepper.G⁻, 
    T = (T_field_G⁻),
    S = (S_field_G⁻),
    u = (u_field_G⁻),
    v = (v_field_G⁻),       
    U = (U_field_G⁻),
    V = (V_field_G⁻),
    e = (e_field_G⁻))         

    for f in ocean.model.timestepper.Gⁿ
        Oceananigans.ImmersedBoundaries.mask_immersed_field!(f)
     end
     
     for f in ocean.model.timestepper.G⁻
        Oceananigans.ImmersedBoundaries.mask_immersed_field!(f)
     end

    u, v, w = ocean.model.velocities
    T, S, e = ocean.model.tracers

    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    erange = (maximum((e)), minimum((e)))

    umax = (maximum(abs, (u)),
            maximum(abs, (v)),
            maximum(abs, (w)))

    @info Trange, Srange, erange, umax
    
    if checkpoint_type == "last"
        @info "Restarting from iteration " * string(maximum(restart_numbers))
    elseif checkpoint_type == "first"
        @info "Restarting from iteration " * string(minimum(restart_numbers))
    end

    simulation.Δt = 12minutes 
    simulation.stop_time = target_time

    run!(simulation)
else
    @info "Running simulation"

    run!(simulation)

    simulation.Δt = 12minutes 
    simulation.stop_time = target_time

    run!(simulation)
end