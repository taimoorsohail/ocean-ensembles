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

checkpoint_timer = (365/2)days
checkpoint_intervals = TimeInterval(checkpoint_timer)
# target_time_input = 14
# target_time = target_time_input * checkpoint_timer
arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)

# if isempty(ARGS)
#     println("No target time provided. Please enter target time:")
#     target_time_input = readline()
#     target_time = parse(Int, target_time_input) * checkpoint_timer
# else
#     @show ARGS
#     target_time = checkpoint_timer*parse(Int,ARGS[4]) * 2 + 22days
# end
# @info target_time
# checkpoint_type = "none" # "none", "last", "first"
## Argument is provided by the submission script!

# if isempty(ARGS)
#     println("No arguments provided. Please enter architecture (CPU/GPU):")
#     arch_input = readline()
#     if arch_input == "GPU"
#         arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
#     elseif arch_input == "CPU"
#         arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
#     else
#         throw(ArgumentError("Invalid architecture. Must be 'CPU' or 'GPU'."))
#     end
# elseif ARGS[2] == "GPU" 
#     arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
# elseif ARGS[2] == "CPU"
#     arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
# else
#     throw(ArgumentError("Architecture must be provided in the format julia --project example_script.jl --arch GPU"))
# end    

total_ranks = MPI.Comm_size(MPI.COMM_WORLD)
localrank = Integer(arch.local_rank)
@info "Using architecture: " * string(arch)
#=
restartfiles = glob("ocean_checkpointer_clock_iteration*rank$(localrank)*", output_path)

# Extract the numeric suffix from each filename
restart_numbers = map(f -> parse(Int, match(r"ocean_checkpointer_clock_iteration(\d+)", basename(f)).captures[1]), restartfiles)
restart_numbers = sort(restart_numbers)

if !isempty(restart_numbers) && maximum(restart_numbers) != 0 && checkpoint_type != "none"
    # Extract the numeric suffix from each filename

    # Get the file with the maximum number
    if checkpoint_type == "last"
        clock_vars = jldopen(output_path * "ocean_checkpointer_clock_iteration" * string(maximum(restart_numbers)) * "_rank$(localrank).jld2")
    elseif checkpoint_type == "first"
        clock_vars = jldopen(output_path * "ocean_checkpointer_clock_iteration" * string(minimum(restart_numbers)) * "_rank$(localrank).jld2")
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
=#
# if time_checkpoint == target_time
#     error("Terminating simulation at target time.")
# end

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

Nx = Integer(360/4)
Ny = Integer(180/4)
ny = Ny/total_ranks
Nz = Integer(75)

@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialDiscretization(Nz, depth, 0, mutable=true) # IMPORTANT: WE NEED TO ACCOUNT FOR THIS

const z_surf = z_faces.cᵃᵃᶠ(Nz)
@info "Top grid cell is " * string(abs(round(z_surf))) * "m thick"

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
                                major_basins = 4)

# Manually masking Black Sea, Caspian Sea and blasting open the Baltic Sea
#=
@info "Applying bathymetry masks"
# Black + Caspian Seas
xs1 = [755, 1010, 1010, 755]
# We split the y-polygon because it stretches across two ranks [unique to 4 rank run :(]
ys1_1 = [800,  790,  809,  809]
ys1_2 = [810,  810,  920,  920]

# Baltic Sea / Danish Straits
xs2 = [679, 670-9, 679, 688+9]
ys2 = [872-10, 875+6, 882+5, 875+6]

nys1_1 = ys1_1/ny
ys1_1_floor = floor.(nys1_1)

nys1_2 = ys1_2/ny
ys1_2_floor = floor.(nys1_2)

nys2 = ys2/ny
ys2_floor = floor.(nys2)

length(unique(ys1_1_floor)) == 1 || error("Mask must all lie in the same rank! Currently they lie in ranks $(unique(ys1_1_floor))")

length(unique(ys1_2_floor)) == 1 || error("Mask must all lie in the same rank! Currently they lie in ranks $(unique(ys1_2_floor))")

length(unique(ys2_floor)) == 1 || error("Mask must all lie in the same rank! Currently they lie in ranks $(unique(ys2_floor))")

if localrank == Integer(unique(ys1_1_floor)[1])
    bh = interior(bottom_height)[:,:,1]
    ys1_1 = [ys1_1[1], ys1_1[2], ys1_1[3]+2, ys1_1[4]+2]
    mask_blacksea_caspian_1 = section_mask(xs1, Int.(ys1_1 .- ny * localrank), ones(length(xs1)), underlying_grid)
    idx = (CuArray(mask_blacksea_caspian_1) .== 1) .& (bh .<= 0)
    bh[idx] .= 0;
    interior(bottom_height) .= bh
end 

if localrank == Integer(unique(ys1_2_floor)[1])
    bh = interior(bottom_height)[:,:,1]
    mask_blacksea_caspian_2 = section_mask(xs1, Int.(ys1_2 .- ny * localrank), ones(length(xs1)), underlying_grid)
    idx = (CuArray(mask_blacksea_caspian_2) .== 1) .& (bh .<= 0)
    bh[idx] .= 0;
    interior(bottom_height) .= bh
end
if localrank == Integer(unique(ys2_floor)[1])
    bh = interior(bottom_height)[:,:,1]
    mask_danish_strait = section_mask(xs2, Int.(ys2 .- ny * localrank), ones(length(xs2)).*2, underlying_grid)
    idx = (CuArray(mask_danish_strait) .== 2) .& (bh .>= 0) .& (bh .<= 3)
    bh[idx] .= -10;
    interior(bottom_height) .= bh
end
=#
@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

### Restoring

# We include surface salinity restoring to a predetermined dataset.

@info "Defining restoring rate"

restoring_rate  = 1 / 30days
@inline mask(x, y, z, t) = z ≥ z_surf - 1

FS = DatasetRestoring(salinity, grid; mask, rate=restoring_rate, time_indices_in_memory = 10)
forcing = (; S=FS)

# ### Closures
# We include a Gent-McWilliam isopycnal diffusivity as a parameterization for the mesoscale
# eddy fluxes. For vertical mixing at the upper-ocean boundary layer we include the CATKE
# parameterization. We also include some explicit horizontal diffusivity.

@info "Defining closures"

catke_closure = ClimaOcean.Oceans.default_ocean_closure()  #RiBasedVerticalDiffusivity()#
closure = (catke_closure, VerticalScalarDiffusivity(κ=1e-5, ν=1e-4))

# ### Ocean simulation
# Now we bring everything together to construct the ocean simulation.
# We use a split-explicit timestepping with 30 substeps for the barotropic
# mode.

@info "Defining free surface"
# output number of substeps
# free_surface = SplitExplicitFreeSurface(grid; cfl=0.7, fixed_Δt=12minutes)

free_surface = SplitExplicitFreeSurface(grid; substeps=60)

momentum_advection = WENOVectorInvariant()
tracer_advection   = WENO(order = 7)

@info "Defining ocean model"

@time ocean = ocean_simulation(grid; Δt=1minutes,
                         momentum_advection,
                         tracer_advection,
                         timestepper = :SplitRungeKutta3,
                         free_surface,
                         forcing = forcing,
                         radiative_forcing = nothing,
                         closure = catke_closure)

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

simulation = Simulation(coupled_model; Δt=12minutes, stop_iteration=5)

import Oceananigans.Diagnostics: CFL
(c::CFL)(sim::Simulation) = c(sim.model)
(c::CFL)(sim::Simulation{<:ClimaOcean.OceanSeaIceModels.OceanSeaIceModel}) = c(sim.model.ocean.model)

#=
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
=#
# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = IterationInterval(1)

function progress(sim)
    η = sim.model.ocean.model.free_surface.displacement
    u, v, w = sim.model.ocean.model.velocities
    T, S = sim.model.ocean.model.tracers
    # cfl = AdvectiveCFL(sim.Δt)

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
    msg8 = @sprintf("SYPD: %.2f \n", (10*sim.Δt)/step_time/365)
    # msg5 = @sprintf("CFL: %.2f \n", getfield(cfl, 1))

    @info msg1 * msg2 * msg3 * msg4 * msg6 * msg7 * msg8

    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, callback_interval)

################################### START OUTPUTTING ######################################
#=

iteration_number = string(Oceananigans.iteration(simulation))

tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)

# surface_height = (; surface_height = ocean.model.free_surface.displacement)
# surface_forcing = (; T_surf = ocean.model.tracers.T.boundary_conditions.top.condition, 
#                     S_surf = ocean.model.tracers.S.boundary_conditions.top.condition)

# outputs_surf = merge(surface_height, surface_forcing)

@info "Defining total integral outputs"

tot_integral = Symbol[]
tot_integral_outputs = Field[]
tot_integral_volume_symbols = Symbol[]
tot_integral_volumes = Field[]

# surf_integral = Symbol[]
# surf_integral_outputs = Field[]
# surf_integral_volume_symbols = Symbol[]
# surf_integral_volumes = Field[]

vert_integral = Symbol[]
vert_integral_outputs = Field[]
vert_integral_volume_symbols = Symbol[]
vert_integral_volumes = Field[]

for key in keys(outputs)
    @show key
    f = outputs[key]
    f_copy = deepcopy(f)
    f_tot = Field(Integral(f, dims = (1,2,3)))
    f_tot_V = Field(Integral(set!(f_copy, 1), dims = (1,2,3)))
    f_vert = Field(Integral(f, dims = (1,2)))
    f_vert_V = Field(Integral(set!(f_copy, 1), dims = (1,2)))

    push!(tot_integral_outputs, f_tot)
    push!(tot_integral, Symbol(key, "_totintegral"))
    push!(tot_integral_volume_symbols, Symbol(key, "_totintegral_volume"))
    push!(tot_integral_volumes, f_tot_V)

    push!(vert_integral_outputs, f_vert)
    push!(vert_integral, Symbol(key, "_vertintegral"))
    push!(vert_integral_volume_symbols, Symbol(key, "_vertintegral_volume"))
    push!(vert_integral_volumes, f_vert_V)
end

# for key in keys(surface_forcing)
#     f_surf = surface_forcing[key]
#     f_surf_copy = deepcopy(f_surf)
#     surf_tot = Field(Integral(f_surf, dims = (1,2,3)))
#     surf_tot_V = Field(Integral(set!(f_surf_copy, 1), dims = (1,2,3)))
#     push!(surf_integral_outputs, surf_tot)
#     push!(surf_integral, Symbol(key, "_surfintegral"))
#     push!(surf_integral_volume_symbols, Symbol(key, "_surfintegral_volume"))
#     push!(surf_integral_volumes, surf_tot_V)
# end

cumulative_tuple = NamedTuple{Tuple(tot_integral)}(Tuple(tot_integral_outputs))
# cumulative_surf_tuple = NamedTuple{Tuple(surf_integral)}(Tuple(surf_integral_outputs))
cumulative_vert_tuple = NamedTuple{Tuple(vert_integral)}(Tuple(vert_integral_outputs))

cumulative_tuple_vol = NamedTuple{Tuple(tot_integral_volume_symbols)}(Tuple(tot_integral_volumes))
# cumulative_surf_tuple_vol = NamedTuple{Tuple(surf_integral_volume_symbols)}(Tuple(surf_integral_volumes))
cumulative_vert_tuple_vol = NamedTuple{Tuple(vert_integral_volume_symbols)}(Tuple(vert_integral_volumes))

global_outputs = merge(cumulative_tuple, cumulative_vert_tuple, cumulative_tuple_vol, cumulative_vert_tuple_vol)

@info "Defining slice outputs"

depths = [0,-100, -500, -1000, -2000]

symbols_slice = Symbol[]  # empty vector to store symbols

for (ind, depth) in enumerate(depths)
    pln, ind_pln =  findmin(abs.(grid.z.cᵃᵃᶜ[1:Nz] .- depths[ind]))
    slice_level = ind_pln
    push!(symbols_slice, Symbol("plane$(abs(round(slice_level, digits=1)))"))
    @show slice_level
    @time ocean.output_writers[symbols_slice[ind]] = JLD2Writer(ocean.model, outputs;
                                                dir = output_path,
                                                schedule = AveragedTimeInterval((365/12)days),
                                                filename = "global_" * string(Integer(round(slice_level))) * "_fields_sxtdeg_RYF_iteration" * iteration_number,
                                                indices = (:, :, ind_pln),
                                                with_halos = false,
                                                including = [:grid, :coriolis, :buoyancy, :closure],
                                                overwrite_existing = true,
                                                array_type = Array{Float32})

end

@time ocean.output_writers[:SSH] = JLD2Writer(ocean.model, outputs_surf;
                                            dir = output_path,
                                            schedule = AveragedTimeInterval((365/12)days),
                                            filename = "global_forcing_fields_sxtdeg_RYF_iteration" * iteration_number,
                                            including = [:grid, :coriolis, :buoyancy, :closure],
                                            with_halos = false,
                                            overwrite_existing = true,
                                            array_type = Array{Float32})


@time ocean.output_writers[:integral] = JLD2Writer(ocean.model, global_outputs;
                                            dir = output_path,
                                            schedule = AveragedTimeInterval((365/48)days),
                                            filename = "global_tot_integrals_sxtdeg_RYF_iteration" * iteration_number,
                                            overwrite_existing = true)

################################### END OUTPUTTING ######################################

################################### START CHECKPOINTING ######################################

@info "Saving restarts"

# function save_restart(sim)
#     localrank = MPI.Comm_rank(MPI.COMM_WORLD)
#     jldsave(output_path * "ocean_checkpointer_clock_iteration" * string(sim.model.clock.iteration) * "_rank$(localrank).jld2";
#     clock = sim.model.ocean.model.clock)
# end

# ocean_checkpointer_tracers = merge(
#     ocean.model.velocities,
#     ocean.model.tracers,
#     ocean.model.free_surface.barotropic_velocities,
#     (; η = simulation.model.ocean.model.free_surface.displacement)
# )
# sea_ice_checkpointer_tracers = merge(  
#                                 (h = sea_ice.model.ice_thickness,
#                                 ℵ = sea_ice.model.ice_concentration,
#                                 Tu = sea_ice.model.ice_thermodynamics.top_surface_temperature,
#                                 Gʰ = sea_ice.model.ice_thermodynamics.thermodynamic_tendency),
#                                 sea_ice.model.dynamics.auxiliaries.fields, 
#                                 sea_ice.model.velocities)

# @time ocean.output_writers[:checkpointer_ocean] = JLD2Writer(ocean.model, ocean_checkpointer_tracers;
#                                             dir = output_path,
#                                             schedule =  TimeInterval((365/2)days),
#                                             filename = "ocean_checkpointer_vars_iteration" * iteration_number,
#                                             with_halos = false,
#                                             including = [:grid, :coriolis, :buoyancy, :closure],
#                                             overwrite_existing = true)

# @time sea_ice.output_writers[:checkpointer_sea_ice] = JLD2Writer(ocean.model, sea_ice_checkpointer_tracers;
#                                             dir = output_path,
#                                             schedule = TimeInterval((365/2)days),
#                                             filename = "sea_ice_checkpointer_vars_iteration" * iteration_number,
#                                             with_halos = false,
#                                             including = [:grid],
#                                             overwrite_existing = true)

# add_callback!(simulation, save_restart, TimeInterval((365/2)days))
=#

@time simulation.output_writers[:checkpointer] = Checkpointer(coupled_model, 
                                                              schedule=IterationInterval(1),  
                                                              dir = output_path, 
                                                              prefix="RYF_sxtdeg_checkpoint_CPU_rank$localrank",
                                                              overwrite_existing = true)

@info "Running Simulation"

simulation.stop_iteration = 140
run!(simulation, pickup=true)

#=
# schedule=WallTimeInterval(30minute)
################################### END CHECKPOINTING ######################################

restartfiles = glob("ocean_checkpointer_vars_iteration*rank$(localrank)*", output_path)

restart_numbers = map(f -> parse(Int, match(r"ocean_checkpointer_vars_iteration(\d+)", basename(f)).captures[1]), restartfiles)
restart_numbers = sort(restart_numbers)

if !isempty(restart_numbers) && maximum(restart_numbers) != 0 && checkpoint_type != "none"
    if checkpoint_type == "last"
        @info "Restarting from last checkpoint"
        ocean_fields_loaded = jldopen(output_path * "ocean_checkpointer_vars_iteration" * string(restart_numbers[end-1]) * "_rank$(localrank).jld2")
        seaice_fields_loaded = jldopen(output_path * "sea_ice_checkpointer_vars_iteration" * string(restart_numbers[end-1]) * "_rank$(localrank).jld2")

    elseif checkpoint_type == "first"
        @info "Restarting from first checkpoint"
        ocean_fields_loaded = jldopen(output_path * "ocean_checkpointer_vars_iteration" * string(restart_numbers[1]) * "_rank$(localrank).jld2")
        seaice_fields_loaded = jldopen(output_path * "sea_ice_checkpointer_vars_iteration" * string(restart_numbers[1]) * "_rank$(localrank).jld2")
    end

    times = parse.(Int, keys(ocean_fields_loaded["timeseries/t"]))
    if checkpoint_type == "last"
        final_time = string(maximum(times))
    elseif checkpoint_type == "first"
        final_time = string(minimum(times))
    end

    @info "Restarting from checkpoint at iteration " * final_time

    T_field = ocean_fields_loaded["timeseries/T/" * final_time]
    S_field = ocean_fields_loaded["timeseries/S/" * final_time]
    e_field = ocean_fields_loaded["timeseries/e/" * final_time]
    u_field = ocean_fields_loaded["timeseries/u/" * final_time]
    v_field = ocean_fields_loaded["timeseries/v/" * final_time]
    w_field = ocean_fields_loaded["timeseries/w/" * final_time]
    η_field = ocean_fields_loaded["timeseries/η/" * final_time]
    U_field = ocean_fields_loaded["timeseries/U/" * final_time]
    V_field = ocean_fields_loaded["timeseries/V/" * final_time]

    h_field = seaice_fields_loaded["timeseries/h/" * final_time]
    ℵ_field = seaice_fields_loaded["timeseries/ℵ/" * final_time]
    σ₁₁_field =  seaice_fields_loaded["timeseries/σ₁₁/" * final_time]
    σ₂₂_field =  seaice_fields_loaded["timeseries/σ₂₂/" * final_time]
    σ₁₂_field =  seaice_fields_loaded["timeseries/σ₁₂/" * final_time]
    Tu_field = seaice_fields_loaded["timeseries/Tu/" * final_time]
    Gʰ_field = seaice_fields_loaded["timeseries/Gʰ/" * final_time]
    u_ice_field = seaice_fields_loaded["timeseries/u/" * final_time]
    v_ice_field = seaice_fields_loaded["timeseries/v/" * final_time]

    close(seaice_fields_loaded)
    close(ocean_fields_loaded)

    set!(ocean.model, 
    T = (T_field),
    S = (S_field),
    e = (e_field),
    u = (u_field),
    v = (v_field),
    w = (w_field))

    set!(ocean.model.free_surface.barotropic_velocities,
    U = (U_field),
    V = (V_field))

    set!(ocean.model.free_surface.displacement, η_field)
    
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
    
    @info "Checkpointers detected; advancing dt and running simulation"

    simulation.Δt = 12minutes
    simulation.stop_time = target_time

    run!(simulation)
else
    @info "No checkpointers detected; Running simulation"

    run!(simulation)

    simulation.Δt = 12minutes 
    simulation.stop_time = target_time

    run!(simulation)
end
=#