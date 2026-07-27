using NumericalEarth

using NumericalEarth.EN4
using NumericalEarth.ECCO
using NumericalEarth.DataWrangling.ETOPO
using NumericalEarth.EarthSystemModels.InterfaceComputations: IceBathHeatFlux

using ClimaSeaIce
using ClimaSeaIce.Rheologies: ElastoViscoPlasticRheology
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Operators: Ax, Ay, Az,
                              Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, Δz⁻¹ᶜᶜᶠ, div_xyᶜᶜᶜ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.Fields: ReducedField, interior, ConstantField, ZeroField, OneField
using Oceananigans.ImmersedBoundaries: immersed_cell, peripheral_node
using Oceananigans.Architectures: on_architecture
using Oceananigans.TimeSteppers: VerticallyImplicitTimeDiscretization, AdaptiveVerticallyImplicitDiscretization
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, AdvectiveFormulation

using CFTime
using Dates
using Printf
using Glob
using JLD2

try
    using CUDA
catch err
    @warn "CUDA could not be loaded. GPU memory diagnostics/reclaiming will be skipped." exception=(err, catch_backtrace())
end

const data_path = expanduser("/home/tsohail/uom/ocean-ensembles/data/")
const output_path = expanduser("/home/tsohail/uom/ocean-ensembles/outputs/onedeg_RYF/")
const figdir = expanduser("/home/tsohail/uom/ocean-ensembles/figures/")

const Nx = Integer(360)
const Ny = Integer(180)
const Nz = Integer(75)
const depth = -5500.0
output_depths = [0, -100, -500, -1000, -2000]

checkpoint_interval = TimeInterval((365/12)days)
output_interval = AveragedTimeInterval((365/12)days)
callback_iteration_interval = 500
default_checkpoint_prefix = "RYF_onedeg_checkpoint"

checkpoint_superprefix(prefix) = prefix * "_iteration"

function checkpoint_iteration(filepath, prefix)
    filename = basename(filepath)
    pattern = Regex("^" * checkpoint_superprefix(prefix) * raw"(\d+)(?:_.*)?\.jld2")
    match_data = match(pattern, filename)
    isnothing(match_data) && return nothing
    return parse(Int, match_data.captures[1])
end

function checkpoint_candidates(prefix; dir=output_path)
    pattern = checkpoint_superprefix(prefix) * "*.jld2"
    filepaths = filter(filepath -> !isnothing(checkpoint_iteration(filepath, prefix)), glob(pattern, dir))

    return sort(filepaths; by=filepath -> (stat(filepath).mtime, checkpoint_iteration(filepath, prefix)), rev=true)
end

function valid_checkpoint(filepath)
    try
        jldopen(filepath, "r") do file
            return !isempty(keys(file))
        end
    catch err
        @warn "Skipping invalid checkpoint file" filepath exception=(err, catch_backtrace())
        return false
    end
end

function latest_valid_checkpoint(prefix; dir=output_path)
    for filepath in checkpoint_candidates(prefix; dir)
        valid_checkpoint(filepath) && return filepath
    end

    return nothing
end

function resolve_pickup(pickup, prefix)
    pickup !== true && return pickup

    filepath = latest_valid_checkpoint(prefix)

    if isnothing(filepath)
        @info "No valid checkpoint found. Starting from scratch."
        return false
    end

    iteration = checkpoint_iteration(filepath, prefix)
    @info "Restarting from last valid checkpoint" filepath iteration
    return filepath
end

function gpu_memory_status(prefix="")
    if !isdefined(Main, :CUDA)
        return nothing
    end

    try
        free, total = CUDA.memory_info()
        @info prefix * "GPU memory" free_GiB=round(free / 2^30, digits=2) total_GiB=round(total / 2^30, digits=2)
    catch err
        @warn "Could not query CUDA memory" exception=(err, catch_backtrace())
    end

    return nothing
end

function reclaim_gpu_memory!(; verbose=true)
    verbose && gpu_memory_status("Before reclaim: ")
    if isdefined(Main, :CUDA)
        try
            CUDA.synchronize()
        catch err
            @warn "Could not synchronize CUDA before reclaiming memory" exception=(err, catch_backtrace())
        end
    end

    for _ = 1:5
        GC.gc(true)
    end

    if isdefined(Main, :CUDA)
        try
            CUDA.reclaim()
        catch err
            @warn "Could not reclaim CUDA memory" exception=(err, catch_backtrace())
        end
    end

    verbose && gpu_memory_status("After reclaim: ")
    return nothing
end


function ryf_dates()
    return vcat(collect(DateTime(1991, 1, 1):Month(1):DateTime(1991, 4, 1)),
                collect(DateTime(1990, 5, 1):Month(1):DateTime(1990, 12, 1)))
end
function ecco_dates()
    return collect(DateTime(1993, 1, 1):Month(1):DateTime(1993, 12, 1))
end

function download_input_data!(dates, dataset)
    @info "Downloading/checking input data"
    @info "We download the 1990-1991 data for an RYF implementation"

    temperature = Metadata(:temperature; dates, dataset=dataset, dir=data_path)
    salinity = Metadata(:salinity; dates, dataset=dataset, dir=data_path)

    ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir=data_path)

    return (; temperature, salinity, ETOPOmetadata)
end

function build_grid(arch, bathymetry_metadata; halo=(7,7,7))
    @info "Defining vertical z faces"
    z = ExponentialDiscretization(Nz, depth, 0, mutable=true)

    @info "Top grid cell is $(Oceananigans.Utils.prettysummary(abs(z.cᵃᵃᶠ[Nz]))) m thick"
    @info "Grid dimensions: Nx = $Nx, Ny = $Ny, Nz = $Nz"

    @info "Defining tripolar grid"
    underlying_grid = TripolarGrid(arch;
                                   size=(Nx, Ny, Nz),
                                   z,
                                   halo,
                                   fold_topology=RightFaceFolded)

    @info "Defining bottom bathymetry"
    @time bottom_height = regrid_bathymetry(underlying_grid, bathymetry_metadata;
                                            minimum_depth=15,
                                            interpolation_passes=25,
                                            major_basins=2)

    @time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

    return grid
end

@inline function advective_inverse_timescale_componentsᶜᶜᶜ(i, j, k, grid, u, v, w)
    inverse_timescale_u = abs(u[i, j, k]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    inverse_timescale_v = abs(v[i, j, k]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)
    inverse_timescale_w = abs(w[i, j, k]) * Δz⁻¹ᶜᶜᶠ(i, j, k, grid)

    return inverse_timescale_u, inverse_timescale_v, inverse_timescale_w
end

@inline function advective_inverse_timescaleᶜᶜᶜ(i, j, k, grid, u, v, w)
    inverse_timescale_u, inverse_timescale_v, inverse_timescale_w =
        advective_inverse_timescale_componentsᶜᶜᶜ(i, j, k, grid, u, v, w)

    inverse_timescale = inverse_timescale_u + inverse_timescale_v + inverse_timescale_w
    return ifelse(isfinite(inverse_timescale), inverse_timescale, -Inf)
end

function maybe_allow_scalar(f)
    return f()
end

if isdefined(Main, :CUDA)
    @eval function maybe_allow_scalar(f)
        return CUDA.@allowscalar f()
    end
end

dominant_cfl_component(cfl_u, cfl_v, cfl_w) =
    cfl_u >= cfl_v && cfl_u >= cfl_w ? :u :
    cfl_v >= cfl_w ? :v : :w

longitude_value(longitudes, i, j) =
    ndims(longitudes) == 1 ? longitudes[i] : longitudes[i, j]

latitude_value(latitudes, i, j) =
    ndims(latitudes) == 1 ? latitudes[j] : latitudes[i, j]

depth_value(depths, i, j, k) =
    ndims(depths) == 1 ? depths[k] : depths[i, j, k]

function host_interior(field)
    return interior(field)
end

if isdefined(Main, :CUDA)
    @eval function host_interior(field)
        interior_field = interior(field)
        if parent(interior_field) isa CUDA.CuArray
            return Array(interior_field)
        else
            return interior_field
        end
    end
end

function findmax_interior_field(field)
    return findmax(host_interior(field))
end

function align_checkpoint_fs(grid, arch, inputs, free_surface::SplitExplicitFreeSurface, checkpoint_prefix)
    @info "Reading checkpoint for free surface alignment"
    filepath = latest_valid_checkpoint(checkpoint_prefix)

    if isnothing(filepath)
        @info "No valid checkpoint found; skipping free-surface alignment."
        return nothing
    end

    jldopen(filepath, "r+") do data
        keys_chkpt = "simulation/model/ocean/model/free_surface/displacement/data"
        checkpoint_halo_size = (Integer((size(data[keys_chkpt])[1]-Nx)/2), Integer((size(data[keys_chkpt])[2]-Ny)/2), size(data[keys_chkpt])[3])
        target_halo_size = (grid.Hx, length(free_surface.substepping.averaging_weights)+2, 1)

        if checkpoint_halo_size != target_halo_size
            @info "Replacing checkpoint free surface with new grid halo size" checkpoint_halo_size target_halo_size
            new_grid = build_grid(arch, inputs.ETOPOmetadata; halo=target_halo_size)
            checkpoint_grid = build_grid(arch, inputs.ETOPOmetadata; halo=checkpoint_halo_size)
            cpu_grid = build_grid(CPU(), inputs.ETOPOmetadata; halo=target_halo_size)

            U_checkpoint = Field{Face, Center, Nothing}(checkpoint_grid)
            V_checkpoint = Field{Center, Face, Nothing}(checkpoint_grid)
            eta_checkpoint = Field{Center, Center, Nothing}(checkpoint_grid)

            parent(U_checkpoint) .= on_architecture(arch, data["simulation/model/ocean/model/free_surface/barotropic_velocities/U/data"])
            parent(V_checkpoint) .= on_architecture(arch, data["simulation/model/ocean/model/free_surface/barotropic_velocities/V/data"])
            parent(eta_checkpoint) .= on_architecture(arch, data["simulation/model/ocean/model/free_surface/displacement/data"])

            U_new = Field{Face, Center, Nothing}(new_grid)
            V_new = Field{Center, Face, Nothing}(new_grid)
            eta_new = Field{Center, Center, Nothing}(new_grid)

            set!(U_new, U_checkpoint)
            set!(V_new, V_checkpoint)
            set!(eta_new, eta_checkpoint)
            Oceananigans.BoundaryConditions.fill_halo_regions!(eta_new)

            U_cpu = Field{Face, Center, Nothing}(cpu_grid)
            V_cpu = Field{Center, Face, Nothing}(cpu_grid)
            eta_cpu = Field{Center, Center, Nothing}(cpu_grid)

            parent(U_cpu) .= on_architecture(CPU(), parent(U_new))
            parent(V_cpu) .= on_architecture(CPU(), parent(V_new))
            parent(eta_cpu) .= on_architecture(CPU(), parent(eta_new))
            delete!(data, "simulation/model/ocean/model/free_surface/barotropic_velocities/U/data")
            delete!(data, "simulation/model/ocean/model/free_surface/barotropic_velocities/V/data")
            delete!(data, "simulation/model/ocean/model/free_surface/displacement/data")

            data["simulation/model/ocean/model/free_surface/barotropic_velocities/U/data"] = parent(U_cpu)
            data["simulation/model/ocean/model/free_surface/barotropic_velocities/V/data"] = parent(V_cpu)
            data["simulation/model/ocean/model/free_surface/displacement/data"] = parent(eta_cpu)
        end
    end

    return nothing
end

# function compute_mht(simulation)
#     esm = simulation.model
#     arch = esm.ocean.model.grid.architecture
#     z = ExponentialDiscretization(Nz, depth, 0, mutable=true)
#     destination_grid = LatitudeLongitudeGrid(arch; size = (360, 180, Nz), halo = (5, 5, 4), z, longitude = (0, 360), latitude = (-89, 89))
#     mht = meridional_heat_transport(esm, TendencyMethod(); destination_grid=destination_grid)
#     return mht
# end

# function compute_TSdiagram(simulation; T_bins = 0:0.5:30, S_bins = 30:0.5:40)
#     ocean_model = simulation.model.ocean.model
#     T, S = ocean_model.tracers
#     h = Histogram((T=T, S=S), bins=(S=S_bins, T=T_bins), weights = :count, method = :integral, dims = (1, 2, 3)) |> Field
#     return h
# end

# function compute_streamfunction(simulation; x_bins = 0:0.5:30, y_bins = 30:0.5:40)
#     ocean_model = simulation.model.ocean.model
#     T, S = ocean_model.tracers
#     h = Histogram((T=T, S=S), bins=(S=S_bins, T=T_bins), weights = :count, method = :integral, dims = (1, 2, 3)) |> Field
#     return h
# end


function add_progress_callback!(simulation; callback_iteration_interval = callback_iteration_interval)
    start_wall_time = Ref(time_ns())
    wall_time = Ref(time_ns())
    callback_interval = IterationInterval(callback_iteration_interval)

    function progress(sim)
        η = sim.model.ocean.model.free_surface.displacement
        u, v, w = sim.model.ocean.model.velocities
        T, S = sim.model.ocean.model.tracers
        iteration = Oceananigans.iteration(sim)

        # The CFL hotspot and dominant-term diagnostics were helpful for debugging,
        # but they add an expensive full-field search and extra reductions/copies.
        # Keep only the aggregate advective CFL in routine progress logging.
        advective_cfl = AdvectiveCFL(sim.Δt)(sim.model.ocean.model)
        # diffusive_cfl = DiffusiveCFL(sim.Δt)(sim.model.ocean.model)
        
        Trange = (maximum(T), minimum(T))
        Srange = (maximum(S), minimum(S))
        ηrange = (maximum(η), minimum(η))

        umax = (maximum(abs, u),
                maximum(abs, v),
                maximum(abs, w))

        current_wall_time = time_ns()
        step_time = 1e-9 * (current_wall_time - wall_time[])
        wall_progress = 1e-9 * (current_wall_time - start_wall_time[])

        msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), iteration, prettytime(sim.Δt))
        msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
        msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Trange...)
        msg4 = @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Srange...)
        msg5 = @sprintf("extrema(η): (%.2f, %.2f) m, ", ηrange...)
        msg6 = @sprintf("wall time: %s\n", prettytime(step_time))
        msg7 = @sprintf("elapsed wall time: %s\n", prettytime(wall_progress))
        msg8 = @sprintf("SYPD: %.2f\n", (callback_iteration_interval * sim.Δt) / step_time / 365)
        msg9 = @sprintf("advective_cfl: %.2f\n", advective_cfl)
        @info msg1 * msg2 * msg3 * msg4 * msg5 * msg6 * msg7 * msg8 * msg9
        wall_time[] = current_wall_time
        return nothing
    end

    add_callback!(simulation, progress, callback_interval)
    return nothing
end
function build_global_outputs(ocean, grid)
    outputs = merge(ocean.model.tracers, ocean.model.velocities)

    tot_integral = Symbol[]
    tot_integral_outputs = Field[]
    vert_integral = Symbol[]
    vert_integral_outputs = Field[]

    for key in keys(outputs)
        f = outputs[key]

        push!(tot_integral_outputs, Field(Integral(f, dims=(1, 2, 3))))
        push!(tot_integral, Symbol(key, "_totintegral"))

        push!(vert_integral_outputs, Field(Integral(f, dims=(1, 2))))
        push!(vert_integral, Symbol(key, "_vertintegral"))
    end

    V_ccc = KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.Vᶜᶜᶜ, grid)
    V_fcc = KernelFunctionOperation{Face, Center, Center}(Oceananigans.Operators.Vᶠᶜᶜ, grid)
    V_cfc = KernelFunctionOperation{Center, Face, Center}(Oceananigans.Operators.Vᶜᶠᶜ, grid)

    tot_integral_volumes = [sum(V_ccc, dims=(1, 2, 3)),
                            sum(V_fcc, dims=(1, 2, 3)),
                            sum(V_cfc, dims=(1, 2, 3))]
    tot_integral_volume_symbols = [:total_volume_c, :total_volume_x, :total_volume_y]

    vert_integral_volumes = [sum(V_ccc, dims=(1, 2)),
                             sum(V_fcc, dims=(1, 2)),
                             sum(V_cfc, dims=(1, 2))]
    vert_integral_volume_symbols = [:vert_volume_c, :vert_volume_x, :vert_volume_y]

    cumulative_tuple = NamedTuple{Tuple(tot_integral)}(Tuple(tot_integral_outputs))
    cumulative_vert_tuple = NamedTuple{Tuple(vert_integral)}(Tuple(vert_integral_outputs))
    cumulative_tuple_vol = NamedTuple{Tuple(tot_integral_volume_symbols)}(Tuple(tot_integral_volumes))
    cumulative_vert_tuple_vol = NamedTuple{Tuple(vert_integral_volume_symbols)}(Tuple(vert_integral_volumes))

    return merge(cumulative_tuple, cumulative_vert_tuple, cumulative_tuple_vol, cumulative_vert_tuple_vol)
end

function slice_output_specs(grid)
    specs = []

    for output_depth in output_depths
        _, ind_pln = findmin(abs.(grid.z.cᵃᵃᶜ[1:Nz] .- output_depth))
        key = Symbol("plane$(abs(round(ind_pln, digits=1)))")
        push!(specs, (; key, slice_level=ind_pln, ind_pln))
    end

    return specs
end

const DiagnosticConstantField = Union{ConstantField, ZeroField, OneField}

function add_run_output_writers!(simulation, ocean, grid, run_id)
    run_id_leading = lpad(string(run_id), 4, '0')
    @info "Defining run-dependent output writers for run $run_id_leading"
    sea_ice_model = simulation.model.sea_ice.model

    sea_ice_outputs = (; ice_thickness=sea_ice_model.ice_thickness,
                       ice_concentration=sea_ice_model.ice_concentration)
    # MHT_outputs = (; mht=compute_mht(simulation))

    outputs = merge(ocean.model.tracers, ocean.model.velocities)
    surface_height = (; surface_height=ocean.model.free_surface.displacement)
    # Surface flux diagnostics are bundled with sea-ice state in restart-era runs.
    # `fw_flux` is the outward-positive freshwater-content flux, including
    # explicit surface volume changes and salt-content exchange.
    surface_forcing = (; heat_flux=Field(net_ocean_heat_flux(simulation.model)),
                       fw_flux=Field(net_ocean_freshwater_flux(simulation.model)))

    for spec in slice_output_specs(grid)
        slice_level = spec.slice_level
        @time ocean.output_writers[spec.key] = JLD2Writer(ocean.model, outputs;
                                                          dir=output_path,
                                                          schedule=output_interval,
                                                          filename="global_" * string(Integer(round(slice_level))) * "_fields_onedeg_RYF_run" * run_id_leading,
                                                          indices=(:, :, spec.ind_pln),
                                                          with_halos=false,
                                                          overwrite_existing=true,
                                                          array_type=Array{Float32})
    end

    @time simulation.output_writers[:surface_conditions] = JLD2Writer(simulation.model, merge(surface_forcing, sea_ice_outputs, surface_height);
                                                                  dir=output_path,
                                                                  schedule=output_interval,
                                                                  filename="global_surface_fluxes_onedeg_RYF_run" * run_id_leading,
                                                                  with_halos=false,
                                                                  overwrite_existing=true,
                                                                  array_type=Array{Float32})

    # @time simulation.output_writers[:MHT] = JLD2Writer(simulation.model, MHT_outputs;
    #                                                               dir=output_path,
    #                                                               schedule=output_interval,
    #                                                               filename="global_MHT_onedeg_RYF_run" * run_id_leading,
    #                                                               with_halos=false,
    #                                                               overwrite_existing=true,
    #                                                               array_type=Array{Float32})

    @time ocean.output_writers[:integral] = JLD2Writer(ocean.model, build_global_outputs(ocean, grid);
                                                       dir=output_path,
                                                       schedule=output_interval,
                                                       filename="global_tot_integrals_onedeg_RYF_run" * run_id_leading,
                                                       overwrite_existing=true)

    return nothing
end

function build_simulation(arch, run_id;
                          add_outputs=true,
                          Δt=10minutes,
                          checkpoint_prefix=default_checkpoint_prefix)
    dates = ecco_dates()
    dataset = ECCO4Monthly()
    time_indices_in_memory = 24
    inputs = download_input_data!(dates, dataset)

    @info "Defining grid"
    grid = build_grid(arch, inputs.ETOPOmetadata; halo=(7,7,7))
    z_surf = CUDA.@allowscalar grid.underlying_grid.z.cᵃᵃᶠ[grid.Nz]

    @info "Defining restoring rate"
    restoring_rate = 1 / 30days
    mask(x, y, z, t) = z ≥ z_surf - 1

    # Keep time caches small so pickup does not require a large transient memory budget.
    FS = DatasetRestoring(inputs.salinity, grid; mask, rate=restoring_rate, time_indices_in_memory)
    forcing = (; S = FS)

    @info "Defining closures"
    catke_closure = NumericalEarth.Oceans.default_ocean_closure()


    eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3, skew_flux_formulation=AdvectiveFormulation())
    closure = (catke_closure, eddy_closure)

    @info "Defining free surface"
    free_surface = SplitExplicitFreeSurface(grid; substeps=167)
    momentum_advection = WENOVectorInvariant(time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5))
    tracer_advection = WENO(order=5, time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5))
    sea_ice_advection = WENO(order=5, minimum_buffer_upwind_order=1)

    align_checkpoint_fs(grid, arch, inputs, free_surface, checkpoint_prefix)

    @info "Defining ocean model"
    @time ocean = ocean_simulation(grid; Δt,
                                   momentum_advection,
                                   tracer_advection,
                                   timestepper=:SplitRungeKutta3,
                                   free_surface,
                                   forcing,
                                   closure)

    @info "Initialising with ECCO4"
    set!(ocean.model,
         T=Metadata(:temperature; dates=first(dates), dataset, dir=data_path),
         S=Metadata(:salinity; dates=first(dates), dataset, dir=data_path))

    @info "Creating sea ice model"
    sea_ice = sea_ice_simulation(grid, ocean;
                                 advection = sea_ice_advection)

    dataset_sea_ice = ECCO4Monthly()
    set!(sea_ice.model,
         h=Metadatum(:sea_ice_thickness; dataset=dataset_sea_ice, dir=data_path),
         ℵ=Metadatum(:sea_ice_concentration; dataset=dataset_sea_ice, dir=data_path))

    @info "Defining Atmospheric state"
    radiation = JRA55PrescribedRadiation(arch; time_indices_in_memory)
    atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory)
    land = JRA55PrescribedLand(arch; time_indices_in_memory)

    @info "Defining coupled model"
    @time coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

    simulation = Simulation(coupled_model; Δt)
    add_progress_callback!(simulation)

    if add_outputs
        add_run_output_writers!(simulation, ocean, grid, run_id)
    end

    @time simulation.output_writers[:checkpointer] = Checkpointer(coupled_model,
                                                                  schedule=checkpoint_interval,
                                                                  dir=output_path,
                                                                  prefix=checkpoint_prefix,
                                                                  overwrite_existing=true,
                                                                  cleanup=false)

    return (; simulation, ocean, run_id)
end

function reconcile_pickup_free_surface!(simulation)
    ocean_model = simulation.model.ocean.model
    Oceananigans.Models.HydrostaticFreeSurfaceModels.reconcile_free_surface!(ocean_model.free_surface, ocean_model.grid, ocean_model.velocities)
    return nothing
end

function run_segment!(state;
                      pickup=false,
                      Δt=nothing,
                      stop_time=nothing,
                      stop_iteration=nothing,
                      wall_time_limit=nothing)
    simulation = state.simulation

    if Δt !== nothing
        @info "Updating simulation time step" Δt=prettytime(Δt)
        simulation.Δt = Δt
    end
    if wall_time_limit !== nothing
        @info "Updating simulation wall-time limit" wall_time_limit=prettytime(wall_time_limit)
        simulation.wall_time_limit = wall_time_limit
    end
    if isnothing(stop_time) && isnothing(stop_iteration)
        simulation.stop_time = state.run_id * 12 * (365 / 12)days
    elseif !isnothing(stop_iteration) && isnothing(stop_time)
        simulation.stop_iteration = stop_iteration
    elseif isnothing(stop_iteration) && !isnothing(stop_time)
        simulation.stop_time = stop_time
    elseif !isnothing(stop_iteration) && !isnothing(stop_time)
        error("Only one of stop_time or stop_iteration should be provided")
    end

    resolved_pickup = resolve_pickup(pickup, default_checkpoint_prefix)

    if resolved_pickup isa String
        @info "Restoring checkpoint before run" filepath=resolved_pickup
        set!(simulation; checkpoint=resolved_pickup)
        @info "Reconciling split-explicit free surface after pickup"
        reconcile_pickup_free_surface!(simulation)
        resolved_pickup = false
    end
    @info "Running simulation" state.run_id pickup=resolved_pickup stop_time=prettytime(simulation.stop_time)
    run!(simulation, pickup=resolved_pickup, checkpoint_at_end=true)

    return nothing
end

# To run
# state = build_simulation(GPU(), 1; add_outputs=true, Δt=5minutes)
# run_segment!(state; pickup=false, Δt=5minutes, stop_time = 1days)

# To rerun
# state = nothing
# reclaim_gpu_memory!()
# state = build_simulation!(GPU(), 1; add_outputs=false)
# run_segment!(state; pickup=true, Δt=5minutes, stop_time = 1days)
