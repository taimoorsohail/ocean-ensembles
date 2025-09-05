using MPI
using CUDA
using CUDA: @allowscalar

MPI.Init()
atexit(MPI.Finalize)  

using ClimaOcean

using ClimaOcean.EN4
using ClimaOcean.ECCO
using ClimaOcean.EN4: download_dataset
using ClimaOcean.DataWrangling.ETOPO

using ClimaSeaIce
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Oceananigans.Operators: Ax, Ay, Az, Δz
using Oceananigans.Fields: ReducedField
using Oceananigans.Architectures: on_architecture

using OceanEnsembles

using CFTime
using Dates
using Printf
using Glob 
using JLD2

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

target_time = 365days*600 # 25 years
checkpoint_type = "last" # "none", "last", "first"

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

    iteration_checkpoint = deepcopy(clock_vars["clock"].iteration)
    time_checkpoint = deepcopy(clock_vars["clock"].time)
    last_Δt_checkpoint = deepcopy(clock_vars["clock"].last_Δt)   

    @info "Moving simulation to " * string(iteration_checkpoint) * " iterations"
    @info "Moving simulation to " * string(prettytime(time_checkpoint))
    @info "Moving simulation last_dt to " * string(last_Δt_checkpoint)

    close(clock_vars)
else
    @info "No valid checkpoint found. Starting from scratch."
    time_checkpoint = 0.0
end

if time_checkpoint == target_time
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
z_faces = ExponentialCoordinate(Nz, depth, 0)

const z_surf = z_faces(Nz)

@info "Top grid cell is " * string(abs(round(z_faces(Nz)))) * "m thick"

# z_faces = Oceananigans.Grids.MutableVerticalDiscretization(z_faces)

@info "Defining tripolar grid"

underlying_grid = TripolarGrid(arch;
                               size = (Nx, Ny, Nz),
                               z = z_faces,
                               halo = (7, 7, 7))

@info "Defining bottom bathymetry"

ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir = data_path)
ClimaOcean.DataWrangling.download_dataset(ETOPOmetadata)

@time bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
                                  minimum_depth = 15,
                                  interpolation_passes = 1, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
				                  major_basins = 2)
view(bottom_height, 73:78, 88:89, 1) .= -1000 # open Gibraltar strait

@info "Defining grid"

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

### Restoring

# We include surface salinity restoring to a predetermined dataset.

@info "Defining restoring rate"

restoring_rate  = 1 / 18days
@inline mask(x, y, z, t) = z ≥ z_surf - 1

FS = DatasetRestoring(salinity, grid; mask, rate=restoring_rate, time_indices_in_memory = 10)
forcing = (; S=FS)

# ### Closures
# We include a Gent-McWilliam isopycnal diffusivity as a parameterization for the mesoscale
# eddy fluxes. For vertical mixing at the upper-ocean boundary layer we include the CATKE
# parameterization. We also include some explicit horizontal diffusivity.

@info "Defining closures"

eddy_closure = Oceananigans.TurbulenceClosures.IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3)
catke_closure = ClimaOcean.OceanSimulations.default_ocean_closure()  #RiBasedVerticalDiffusivity()
closure = (catke_closure, VerticalScalarDiffusivity(κ=1e-5, ν=1e-4), eddy_closure)

# ### Ocean simulation
# Now we bring everything together to construct the ocean simulation.
# We use a split-explicit timestepping with 30 substeps for the barotropic
# mode.

@info "Defining free surface"

free_surface = SplitExplicitFreeSurface(grid; cfl=0.7, fixed_Δt=45minutes)
momentum_advection = WENOVectorInvariant(order = 5)
tracer_advection   = WENO(order = 5)

@time ocean = ocean_simulation(grid; Δt=1minutes,
                         momentum_advection,
                         tracer_advection,
                         timestepper = :SplitRungeKutta3,
                         free_surface,
                         forcing = forcing,
                         closure)

# ### Initial condition

# We initialize the ocean from the ECCO state estimate.

@info "Initialising with EN4"

set!(ocean.model, T=Metadata(:temperature; dates=first(dates), dataset = dataset, dir=data_path),
                  S=Metadata(:salinity;    dates=first(dates), dataset = dataset, dir=data_path))

#####
##### A Prognostic Sea-ice model
#####

# Default sea-ice dynamics and salinity coupling are included in the defaults
sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7)) 

set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dataset=ECCO4Monthly(), dir=data_path),
                    ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly(), dir=data_path))

# ### Atmospheric forcing

# We force the simulation with an JRA55-do atmospheric reanalysis.
@info "Defining Atmospheric state"

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(100), include_rivers_and_icebergs=true)

# ### Coupled simulation

# Now we are ready to build the coupled ocean--sea ice model and bring everything
# together into a `simulation`.

# We use a relatively short time step initially and only run for a few days to
# avoid numerical instabilities from the initial "shock" of the adjustment of the
# flow fields.

@info "Defining coupled model"
@time coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

simulation = Simulation(coupled_model; Δt=20minutes, stop_time=60days)

# ### Restarting the simulation
if !isempty(restart_numbers) && maximum(restart_numbers) != 0 && checkpoint_type != "none"
    simulation.model.ocean.model.clock.iteration = iteration_checkpoint
    simulation.model.ocean.model.clock.time = time_checkpoint
    simulation.model.sea_ice.model.clock.iteration = iteration_checkpoint
    simulation.model.sea_ice.model.clock.time = time_checkpoint
    simulation.model.atmosphere.clock.iteration = iteration_checkpoint
    simulation.model.atmosphere.clock.time = time_checkpoint
    simulation.model.clock.iteration = iteration_checkpoint
    simulation.model.clock.time = time_checkpoint
    time_step!(atmosphere, 0)
    simulation.model.atmosphere.clock.iteration -= 1
    simulation.model.ocean.model.clock.last_Δt = last_Δt_checkpoint
    simulation.model.sea_ice.model.clock.last_Δt = last_Δt_checkpoint
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
    T, S = sim.model.ocean.model.tracers

    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    ηrange = (maximum((η)), minimum((η)))

    umax = (maximum(abs, (u)),
            maximum(abs, (v)),
            maximum(abs, (w)))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s,", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Trange...)
    msg4 = @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Srange...)
    msg6 = @sprintf("extrema(η): (%.2f, %.2f) m, ", ηrange...)
    msg7 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg3 * msg4 * msg6 * msg7
    
    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, callback_interval)
checkpoint_intervals(73days)

# #### REGRIDDING ####

# @info "Defining destination underlying grid"
# underlying_destination_grid = LatitudeLongitudeGrid(arch;
#                              size = (Nx, Ny, Nz),
#                              halo = (7, 7, 7),
#                              z = z_faces,
#                              latitude  = (-80, 90),
#                              longitude = (0, 360))

# @info "Interpolating bottom bathymetry"

# @time bottom_height = regrid_bathymetry(underlying_destination_grid, ETOPOmetadata;
#                                   minimum_depth = 15,
#                                   interpolation_passes = 1, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
# 				                  major_basins = 2)
# # view(bottom_height, 73:78, 88:89, 1) .= -1000 # open Gibraltar strait
                                  
# @info "Defining destination grid"

# @time destination_grid = ImmersedBoundaryGrid(underlying_destination_grid, GridFittedBottom(bottom_height); active_cells_map=true)

# destination_field = Field{Center, Center, Center}(destination_grid)
# source_field = Field{Center, Center, Center}(grid)

# @info "Defining regridder weights"
# W_bilin = @allowscalar regridder_weights(source_field, destination_field; method = "bilinear")
# W_cons = @allowscalar regridder_weights(source_field, destination_field; method = "conservative")

# @info "Defining outputs - bilinearly interpolated"

# tracers = ocean.model.tracers
# velocities = ocean.model.velocities

# outputs = merge(tracers, velocities)
# output_names_bilin = Symbol[]
# outputs_bilin = Field[]
# output_names_cons = Symbol[]
# outputs_cons = Field[]

# for key in keys(tracers)
#     @info "Regridding output: " * string(key)
#     f = outputs[key]
#     f_dst_bilin = regrid_tracers!(f, destination_field, W_bilin)                       
#     push!(outputs_bilin, f_dst_bilin)
#     push!(output_names_bilin, Symbol(key, "_bilinear"))
#     f_dst_cons = regrid_tracers!(f, destination_field, W_cons)                       
#     push!(outputs_cons, f_dst_cons)
#     push!(output_names_cons, Symbol(key, "_conservative"))
# end

# #### VERTICAL INTEGRALS ####
# @info "Defining vertical integral outputs"

# vertical_integral_conservative = Symbol[]
# vertical_integral_conservative_outputs = Field[]
# tot_integral_conservative = Symbol[]
# tot_integral_conservative_outputs = Field[]
# avg_conservative = Symbol[]
# avg_conservative_outputs = Field[]

# for key in keys(conservative_tuple)
#     f = conservative_tuple[key]
#     f_int = CumulativeIntegral(f, dims = 3)
#     f_tot = Integral(f, dims = (1,2,3))
#     f_avg = Average(f, dims = (1,2,3))
#     push!(tot_integral_conservative_outputs, f_tot)
#     push!(tot_integral_conservative, Symbol(key, "_totintegral"))
#     push!(avg_conservative_outputs, f_avg)
#     push!(avg_conservative, Symbol(key, "_avg"))
#     push!(vertical_integral_conservative_outputs, f_int)
#     push!(vertical_integral_conservative, Symbol(key, "_cumintegral"))
# end

# bilinear_tuple = NamedTuple{Tuple(output_names_bilin)}(Tuple(outputs_bilin))
# conservative_tuple = NamedTuple{Tuple(output_names_cons)}(Tuple(outputs_cons))
# vertical_integral_tuple = NamedTuple{Tuple(vertical_integral_conservative)}(Tuple(vertical_integral_conservative_outputs))
# cumulative_tuple = NamedTuple{Tuple(tot_integral_conservative)}(Tuple(tot_integral_conservative_outputs))
# average_tuple = NamedTuple{Tuple(avg_conservative)}(Tuple(avg_conservative_outputs))

# global_outputs = merge(cumulative_tuple, average_tuple)
# #### OUTPUTS ####
# iteration_number = string(Oceananigans.iteration(simulation))

# @info "Defining slice outputs"

# depths = [0,-100, -500, -1000, -2000]

# symbols_slice = Symbol[]  # empty vector to store symbols
# symbols_cumint = Symbol[]  # empty vector to store symbols

# for (ind, depth) in enumerate(depths)
#     pln, ind_pln =  findmin(abs.(grid.z.cᵃᵃᶜ[1:Nz] .- depths[ind]))
#     push!(symbols_slice, Symbol("plane_$(abs(round(pln, digits=1)))m"))
#     push!(symbols_cumint, Symbol("integrated_$(abs(round(pln, digits=1)))m"))

#     @time simulation.output_writers[symbols_slice[ind]] = JLD2Writer(ocean.model, conservative_tuple;
#                                                 dir = output_path,
#                                                 schedule = TimeInterval(1days),
#                                                 filename = "global_*" * string(round(pln)) * "m_fields_onedeg_iteration" * iteration_number,
#                                                 indices = (:, :, ind_pln),
#                                                 with_halos = false,
#                                                 overwrite_existing = true,
#                                                 array_type = Array{Float32})

#     @time simulation.output_writers[symbols_cumint[ind]] = JLD2Writer(ocean.model, vertical_integral_tuple;
#                                                 dir = output_path,
#                                                 schedule = TimeInterval(1days),
#                                                 filename = "global_*" * string(round(pln)) * "m_integral_onedeg_iteration" * iteration_number,
#                                                 overwrite_existing = true)
                                            
#     end

# @time simulation.output_writers[:global_diags] = JLD2Writer(ocean.model, global_outputs;
#                                             dir = output_path,
#                                             schedule = TimeInterval(1days),
#                                             filename = "global_tot_integrals_onedeg_iteration" * iteration_number,
#                                             overwrite_existing = true)



#### TRACERS ####

# @info "Defining tracer outputs - conservatively regridded"

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
# tracer_names = Symbol[]
# tracer_outputs = Reduction[]

# @info "Tracers"
# @time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, dims = (3));

# tracer_tuple = NamedTuple{Tuple(tracer_names)}(Tuple(tracer_outputs))


# # for j in 1:3
# #     # @time ocean_tracer_content!(tracer_names, tracer_outputs; dst_field = destination_field, weights = W_cons, outputs=tracers, operator = tracer_volmask[2], dims = (1, 2), condition = masks[j][1], suffix = suffixes[j]*"depth");
# #     # @time ocean_tracer_content!(tracer_names, tracer_outputs; dst_field = destination_field, weights = W_cons, outputs=tracers, operator = tracer_volmask[3], dims = (1, 2, 3), condition = masks[j][1], suffix = suffixes[j]*"tot");
# # end

# @info "Merging tracer tuples"


# #### VELOCITIES ####
# @info "Velocities"

# transport_volmask_operators = [Ax, Ay, Az]
# transport_names = Symbol[]
# transport_outputs = ReducedField[]
# for j in 1:3
#     @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1), condition = masks[j], suffix = suffixes[j]*"zonal")
#     @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1,2), condition = masks[j], suffix = suffixes[j]*"depth")
#     @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1,2,3), condition = masks[j], suffix = suffixes[j]*"tot")
# end

# @info "Merging velocity tuples"

# transport_tuple = NamedTuple{Tuple(transport_names)}(Tuple(transport_outputs))

# @info "Defining output writers"


# @time simulation.output_writers[:transport] = JLD2Writer(ocean.model, transport_tuple;
#                                                           dir = output_path,
#                                                           schedule = TimeInterval(1days),
#                                                           filename = "mass_transport_onedeg_iteration" * iteration_number,
#                                                           overwrite_existing = true)



# @time simulation.output_writers[:ocean_tracer_content] = JLD2Writer(ocean.model, tracer_tuple;
#                                                           dir = output_path,
#                                                           schedule = TimeInterval(1days),
#                                                           filename = "ocean_tracer_content_onedeg_iteration" * iteration_number,
#                                                           overwrite_existing = true)

tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)

tot_integral = Symbol[]
tot_integral_outputs = Field[]
avg = Symbol[]
avg_outputs = Field[]

for key in keys(outputs)
    f = outputs[key]
    f_tot = Field(Integral(f, dims = (1,2,3)))
    f_avg = Field(Average(f, dims = (1,2,3)))
    push!(tot_integral_outputs, f_tot)
    push!(tot_integral, Symbol(key, "_totintegral"))
    push!(avg_outputs, f_avg)
    push!(avg, Symbol(key, "_avg"))
end

cumulative_tuple = NamedTuple{Tuple(tot_integral)}(Tuple(tot_integral_outputs))
average_tuple = NamedTuple{Tuple(avg)}(Tuple(avg_outputs))

global_outputs = merge(cumulative_tuple, average_tuple)

iteration_number = string(Oceananigans.iteration(simulation))

@info "Defining slice outputs"

depths = [0,-100, -500, -1000, -2000]

symbols_slice = Symbol[]  # empty vector to store symbols

for (ind, depth) in enumerate(depths)
    pln, ind_pln =  findmin(abs.(grid.z.cᵃᵃᶜ[1:Nz] .- depths[ind]))
    slice_level = abs(z_faces(ind_pln))
    push!(symbols_slice, Symbol("plane_$(abs(round(slice_level, digits=1)))m"))

    @time simulation.output_writers[symbols_slice[ind]] = JLD2Writer(ocean.model, outputs;
                                                dir = output_path,
                                                schedule = TimeInterval(1days),
                                                filename = "global_" * string(Integer(round(slice_level))) * "m_fields_onedeg_RYF_iteration" * iteration_number,
                                                indices = (:, :, ind_pln),
                                                with_halos = false,
                                                overwrite_existing = true,
                                                array_type = Array{Float32})

end

@time simulation.output_writers[:global_diags] = JLD2Writer(ocean.model, global_outputs;
                                            dir = output_path,
                                            schedule = TimeInterval(1days),
                                            filename = "global_tot_integrals_onedeg_RYF_iteration" * iteration_number,
                                            overwrite_existing = true)


#### CHECKPOINTING ####
# if checkpoint_type != "none"
#     @info "Removing all checkpoints"
#     for f in restartfiles
#         if isfile(f)
#             @info "Removing old restart file: $f"
#             rm(f; force = true)
#         end
#     end
# end

@info "Saving restart"

function save_restart(sim)
    @info @sprintf("Saving checkpoint file")

    jldsave(output_path * "checkpoint_onedeg_iteration" * string(sim.model.clock.iteration) * ".jld2";
    u = on_architecture(CPU(), (sim.model.ocean.model.velocities.u)),
    v = on_architecture(CPU(), (sim.model.ocean.model.velocities.v)),
    w = on_architecture(CPU(), (sim.model.ocean.model.velocities.w)),
    T = on_architecture(CPU(), (sim.model.ocean.model.tracers.T)),
    S = on_architecture(CPU(), (sim.model.ocean.model.tracers.S)),
    e = on_architecture(CPU(), (sim.model.ocean.model.tracers.e)),
    η = on_architecture(CPU(), (sim.model.ocean.model.free_surface.η)),
    U = on_architecture(CPU(), (sim.model.ocean.model.free_surface.barotropic_velocities.U)),
    V = on_architecture(CPU(), (sim.model.ocean.model.free_surface.barotropic_velocities.V)),

    h = on_architecture(CPU(), (sim.model.sea_ice.model.ice_thickness)),
    ℵ = on_architecture(CPU(), (sim.model.sea_ice.model.ice_concentration)),
    σ₁₁ = on_architecture(CPU(), (sim.model.sea_ice.model.dynamics.auxiliaries.fields.σ₁₁)),
    σ₂₂ = on_architecture(CPU(), (sim.model.sea_ice.model.dynamics.auxiliaries.fields.σ₂₂)),
    σ₁₂ = on_architecture(CPU(), (sim.model.sea_ice.model.dynamics.auxiliaries.fields.σ₁₂)),
    Tu = on_architecture(CPU(), (sim.model.sea_ice.model.ice_thermodynamics.top_surface_temperature)),
    Gʰ = on_architecture(CPU(), (sim.model.sea_ice.model.ice_thermodynamics.thermodynamic_tendency)),
    u_ice = on_architecture(CPU(), (sim.model.sea_ice.model.velocities.u)),
    v_ice = on_architecture(CPU(), (sim.model.sea_ice.model.velocities.v)),

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
    η_field = fields_loaded["η"]
    U_field = fields_loaded["U"]
    V_field = fields_loaded["V"]

    h_field = fields_loaded["h"]
    ℵ_field = fields_loaded["ℵ"]
    σ₁₁_field =  fields_loaded["σ₁₁"]
    σ₂₂_field =  fields_loaded["σ₂₂"]
    σ₁₂_field =  fields_loaded["σ₁₂"]
    Tu_field = fields_loaded["Tu"]
    Gʰ_field = fields_loaded["Gʰ"]
    u_ice_field = fields_loaded["u_ice"]
    v_ice_field = fields_loaded["v_ice"]

    close(fields_loaded)

    set!(ocean.model, 
    T = (T_field),
    S = (S_field),
    e = (e_field),
    u = (u_field),
    v = (v_field),
    w = (w_field),
    η = (η_field))

    set!(ocean.model.free_surface.barotropic_velocities,
    U = (U_field),
    V = (V_field))
    
    set!(sea_ice.model, 
    h = (h_field),
    ℵ = (ℵ_field))
    
    set!(sea_ice.model.dynamics.auxiliaries.fields.σ₁₁, σ₁₁_field)
    set!(sea_ice.model.dynamics.auxiliaries.fields.σ₂₂, σ₂₂_field)
    set!(sea_ice.model.dynamics.auxiliaries.fields.σ₁₂, σ₁₂_field)
    set!(sea_ice.model.ice_thermodynamics.top_surface_temperature, Tu_field)
    set!(sea_ice.model.ice_thermodynamics.thermodynamic_tendency, Gʰ_field)
    set!(sea_ice.model.velocities.u, u_ice_field)
    set!(sea_ice.model.velocities.v, v_ice_field)
    
    @info "Running simulation"

    simulation.Δt = 20minutes
    simulation.stop_time = target_time

    run!(simulation)
else
    @info "Running simulation"

    run!(simulation)

    simulation.Δt = 20minutes 
    simulation.stop_time = target_time

    run!(simulation)
end
