using NumericalEarth
using NumericalEarth.DataWrangling: SurfaceFluxRestoring,
                                    ConservativeSurfaceFluxRestoring,
                                    ConservativeSurfaceFluxRestoringCallback

using NumericalEarth.Oceans: river_mouth_vertical_diffusivity
using NumericalEarth.DataWrangling.SeaWiFS: SeaWiFSMonthly
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
using Oceananigans.Operators: Ax, Ay, Az,
                              Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, Δz⁻¹ᶜᶜᶠ, div_xyᶜᶜᶜ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.Fields: ReducedField, interior, ConstantField, ZeroField, OneField
using Oceananigans.ImmersedBoundaries: immersed_cell, peripheral_node
using Oceananigans.Architectures: on_architecture
using Oceananigans.TimeSteppers: VerticallyImplicitTimeDiscretization, AdaptiveVerticallyImplicitDiscretization
using Oceananigans.TurbulenceClosures: compute_closure_fields!
using Oceananigans.OutputWriters: load_checkpoint_state

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
const output_path = expanduser("/home/tsohail/uom/ocean-ensembles/outputs/")
const figdir = expanduser("/home/tsohail/uom/ocean-ensembles/figures/")

const Nx = Integer(360 * 6)
const Ny = Integer(180 * 6)
const Nz = Integer(75)
const depth = -5500.0
output_depths = [0, -100, -500, -1000, -2000]

checkpoint_interval = TimeInterval((365/2)days)
output_interval = AveragedTimeInterval(5days)
callback_iteration_interval = 100
default_checkpoint_prefix = "RYF_sxtdeg_checkpoint"

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
                                            major_basins=1)

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

    unit_ccc = set!(CenterField(grid), 1)
    unit_fcc = set!(XFaceField(grid), 1)
    unit_cfc = set!(YFaceField(grid), 1)

    tot_integral_volumes = [Field(Integral(unit_ccc, dims=(1, 2, 3))),
                            Field(Integral(unit_fcc, dims=(1, 2, 3))),
                            Field(Integral(unit_cfc, dims=(1, 2, 3)))]
    tot_integral_volume_symbols = [:total_volume_c, :total_volume_x, :total_volume_y]

    vert_integral_volumes = [Field(Integral(unit_ccc, dims=(1, 2))),
                             Field(Integral(unit_fcc, dims=(1, 2))),
                             Field(Integral(unit_cfc, dims=(1, 2)))]
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
                                                          filename="global_" * string(Integer(round(slice_level))) * "_fields_sxtdeg_RYF_run" * run_id_leading,
                                                          indices=(:, :, spec.ind_pln),
                                                          with_halos=false,
                                                          overwrite_files=true,
                                                          array_type=Array{Float32})
    end

    @time simulation.output_writers[:surface_conditions] = JLD2Writer(simulation.model, merge(surface_forcing, sea_ice_outputs, surface_height);
                                                                  dir=output_path,
                                                                  schedule=output_interval,
                                                                  filename="global_surface_fluxes_sxtdeg_RYF_run" * run_id_leading,
                                                                  with_halos=false,
                                                                  overwrite_files=true,
                                                                  array_type=Array{Float32})

    # @time simulation.output_writers[:MHT] = JLD2Writer(simulation.model, MHT_outputs;
    #                                                               dir=output_path,
    #                                                               schedule=output_interval,
    #                                                               filename="global_MHT_sxtdeg_RYF_run" * run_id_leading,
    #                                                               with_halos=false,
    #                                                               overwrite_files=true,
    #                                                               array_type=Array{Float32})

    @time ocean.output_writers[:integral] = JLD2Writer(ocean.model, build_global_outputs(ocean, grid);
                                                       dir=output_path,
                                                       schedule=output_interval,
                                                       filename="global_tot_integrals_sxtdeg_RYF_run" * run_id_leading,
                                                       overwrite_files=true)

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
    @info "Defining conservative surface salinity restoring"
    restoring_rate = 1 / 30days

    # Keep time caches small so pickup does not require a large transient memory budget.
    salinity_restoring = DatasetRestoring(inputs.salinity, grid;
                                          rate=restoring_rate,
                                          time_indices_in_memory)
    surface_salinity_restoring = SurfaceFluxRestoring(salinity_restoring)
    conservative_salinity_restoring =
        ConservativeSurfaceFluxRestoring(surface_salinity_restoring, grid)
    additional_surface_fluxes = (; S=conservative_salinity_restoring)

    # Simone's OMIP 1/12-degree runoff settings, with the search reach scaled to our grid.
    maximum_search_radius = max(5, ceil(Int, 3 / ((360 / Nx + 180 / Ny) / 2)))
    @info "Routing RYF river and iceberg discharge"
    land = JRA55PrescribedLand(grid;
                              dataset=RepeatYearJRA55(),
                              time_indices_in_memory,
                              maximum_search_radius,
                              spread_radius=1.2,
                              maximum_spread_cells=nothing)
    @info "Defining closures"
    catke_closure = NumericalEarth.Oceans.default_ocean_closure()
    background_closure = VerticalScalarDiffusivity(κ=1e-5, ν=1e-4)

    river_closure = river_mouth_vertical_diffusivity(grid, land.river_routing;
                                                    river_mouth_diffusivity=0.1,
                                                    river_mouth_mixing_depth=10)
    closure = (catke_closure, background_closure, river_closure)

    @info "Defining free surface"
    free_surface = SplitExplicitFreeSurface(grid; substeps=167)
    momentum_advection = WENOVectorInvariant(time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5))
    tracer_advection = (T = WENO(order=7,
                                   time_discretization=AdaptiveVerticallyImplicitDiscretization(cfl=0.5)),
                        S = WENO(order=7,
                                 bounds=(0, 100),
                                 time_discretization=AdaptiveVerticallyImplicitDiscretization(cfl=0.5)))

    align_checkpoint_fs(grid, arch, inputs, free_surface, checkpoint_prefix)

    @info "Loading monthly SeaWiFS chlorophyll for shortwave absorption"
    chlorophyll_dates = (DateTime(2000, 1, 1), DateTime(2000, 12, 1))
    chlorophyll_metadata = Metadata(:chlorophyll;
                                    dataset=SeaWiFSMonthly(),
                                    dates=chlorophyll_dates,
                                    dir=data_path)
    # Repeat the same twelve monthly fields, as in Simone's OMIP setup.
    chlorophyll = FieldTimeSeries(chlorophyll_metadata, grid)
    radiative_forcing = TwoColorRadiation(grid; chlorophyll)

    @info "Defining ocean model"
    @time ocean = ocean_simulation(grid; Δt,
                                   radiative_forcing,
                                   momentum_advection,
                                   tracer_advection,
                                   timestepper=:SplitRungeKutta3,
                                   free_surface,
                                   additional_surface_fluxes,
                                   closure)

    @info "Initialising with ECCO4"
    set!(ocean.model,
         T=Metadata(:temperature; dates=first(dates), dataset, dir=data_path),
         S=Metadata(:salinity; dates=first(dates), dataset, dir=data_path))

    @info "Creating sea ice model"
    sea_ice = sea_ice_simulation(grid, ocean;
                                 advection = ClimaSeaIce.IncrementalRemapping(),
                                 timestepper = :ForwardEuler)

    dataset_sea_ice = ECCO4Monthly()
    set!(sea_ice.model,
         h=Metadatum(:sea_ice_thickness; dataset=dataset_sea_ice, dir=data_path),
         ℵ=Metadatum(:sea_ice_concentration; dataset=dataset_sea_ice, dir=data_path))

    @info "Defining Atmospheric state"
    radiation = JRA55PrescribedRadiation(arch; time_indices_in_memory)
    atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory)

    @info "Defining coupled model"
    @time coupled_model = OceanSeaIceModel(ocean, sea_ice; land, atmosphere, radiation)

    simulation = Simulation(coupled_model; Δt)

    restoring_callback =
        ConservativeSurfaceFluxRestoringCallback(conservative_salinity_restoring, ocean.model)
    add_callback!(simulation, restoring_callback, IterationInterval(1);
                  name=:conservative_salinity_restoring)

    add_progress_callback!(simulation)

    if add_outputs
        add_run_output_writers!(simulation, ocean, grid, run_id)
    end

    @time simulation.output_writers[:checkpointer] = Checkpointer(coupled_model,
                                                                  schedule=checkpoint_interval,
                                                                  dir=output_path,
                                                                  prefix=checkpoint_prefix,
                                                                  overwrite_files=true,
                                                                  cleanup=false)

    return (; simulation, ocean, run_id)
end

function reset_catke_pickup_memory!(closure_fields::Tuple, closures::Tuple, model)
    for (fields, closure) in zip(closure_fields, closures)
        reset_catke_pickup_memory!(fields, closure, model)
    end

    return nothing
end

function reset_catke_pickup_memory!(closure_fields, closure, model)
    hasproperty(closure_fields, :previous_velocities) || return nothing
    hasproperty(closure_fields, :Jᵇ) || return nothing

    u, v, w = model.velocities
    u⁻ = closure_fields.previous_velocities.u
    v⁻ = closure_fields.previous_velocities.v

    parent(u⁻) .= parent(u)
    parent(v⁻) .= parent(v)
    parent(closure_fields.Jᵇ) .= 0

    @info "Reset CATKE pickup memory" closure=summary(closure)
    return nothing
end

function synchronize_pickup_clock!(component, clock_state)
    hasproperty(component, :clock) || return nothing

    Oceananigans.restore_prognostic_state!(component.clock, clock_state)
    Oceananigans.TimeSteppers.update_state!(component)
    @info "Synchronized pickup component clock" component=summary(component)

    return nothing
end

synchronize_pickup_clock!(::Nothing, clock_state) = nothing

function synchronize_pickup_component_clocks!(model)
    clock_state = Oceananigans.prognostic_state(model.clock)

    synchronize_pickup_clock!(model.radiation, clock_state)
    synchronize_pickup_clock!(model.atmosphere, clock_state)
    synchronize_pickup_clock!(model.land, clock_state)

    return nothing
end

# Equals 1 in ocean cells and 0 in land or sea-floor cells.
@inline ocean_cell_indicator(i, j, k, grid) = ifelse(immersed_cell(i, j, k, grid), 0.0, 1.0)

# A CPU array of size (Nx, Ny, Nz) that is true in ocean cells and false in land or sea-floor cells.
# The work is done on the GPU, then copied to the CPU.
function ocean_cells(grid)
    ocean_indicator = Field(KernelFunctionOperation{Center, Center, Center}(ocean_cell_indicator, grid))
    Oceananigans.Fields.compute!(ocean_indicator)
    return Array(interior(ocean_indicator)) .== 1
end

"""
    checkpoint_matches_grid(grid, filepath)

Return `true` if the checkpoint at `filepath` was saved with the same sea floor as `grid`.

Newer checkpoints save the sea floor (`bottom_height`), so we compare it directly.
Older checkpoints do not, so we look at the saved salinity instead. Land and sea-floor
cells have salinity exactly zero in a checkpoint, so an ocean cell of the current grid
with zero saved salinity was not ocean when the checkpoint was saved.
"""
function checkpoint_matches_grid(grid, filepath)
    ocean_path = "simulation/model/ocean/model"

    return jldopen(filepath, "r") do file
        if haskey(file, ocean_path * "/bottom_height")
            saved_bottom_height = file[ocean_path * "/bottom_height"]
            return saved_bottom_height == Oceananigans.ImmersedBoundaries.bottom_height_checkpoint_state(grid)
        end

        # The saved salinity includes the halo cells around the edges, so cut those off.
        saved_salinity = file[ocean_path * "/tracers/S/data"]
        Nx_grid, Ny_grid, Nz_grid = size(grid)
        Hx, Hy, Hz = (size(saved_salinity) .- (Nx_grid, Ny_grid, Nz_grid)) .÷ 2
        saved_salinity = view(saved_salinity, Hx+1:Hx+Nx_grid, Hy+1:Hy+Ny_grid, Hz+1:Hz+Nz_grid)

        return !any(ocean_cells(grid) .& (saved_salinity .== 0))
    end
end

"""
    fill_new_ocean_cells!(ocean_model; tracer_names=(:T, :S), maximum_passes=1000)

Fix up cells that are ocean in the current grid but were land or sea floor in the
checkpoint. This happens when the sea floor used to build the grid has changed since
the checkpoint was saved. Those cells come back from the checkpoint with zero
temperature and zero salinity, and the large density jump next to real sea water
makes the flow blow up in the first time step.

We find these cells as ocean cells whose salinity is exactly zero. Then we fill them
outward from the old ocean, one layer of cells per pass: each cell takes the average of
its neighbours (east, west, north, south, above, below) that are ocean cells and already
have values. Any cell that still has no filled neighbour after `maximum_passes` takes
the average of the old ocean cells at the same depth level.

Some columns became much deeper. If a column now has at least as many new cells as old
ones, all of its ocean cells are refilled this way, not just the new ones. Otherwise a
thin old layer can sit on top of the new water below it. On the Antarctic shelf that
old layer is often very salty water left behind by sea ice, which is much heavier than
the filled water below it. It sinks violently and the run blows up within about 50 steps.
This throws away a small amount of salt and heat in those columns.

The sea-surface height η needs the same fix. A column that was land in the checkpoint
has η = 0 saved there, while the ocean next to it can sit metres lower. That sudden
step in η pushes water very hard on the first time step. So columns with no old ocean
cell at all get η filled in exactly the same way, treating η as a single level.
Columns that were already ocean keep their own η.

Velocities are left alone. They are already zero on the faces of these cells because
those faces were land in the checkpoint.
"""
function fill_new_ocean_cells!(ocean_model; tracer_names=(:T, :S), maximum_passes=1000)
    grid = ocean_model.grid
    tracers = ocean_model.tracers

    ocean = ocean_cells(grid)

    salinity = Array(interior(tracers.S))
    new_cell = ocean .& (salinity .== 0)
    number_of_new_cells = count(new_cell)

    if number_of_new_cells == 0
        @info "Pickup has no new ocean cells to fill"
        return nothing
    end

    # Columns that gained at least as many new cells as they had old ones get every
    # ocean cell refilled, so a thin old layer of very salty water is not left on top.
    number_of_new_levels = sum(new_cell, dims=3)
    number_of_old_levels = sum(ocean .& .!new_cell, dims=3)
    deepened_column = (number_of_old_levels .> 0) .& (number_of_new_levels .>= number_of_old_levels)
    refill_cell = new_cell .| (ocean .& deepened_column)
    number_of_deepened_columns = count(deepened_column)

    # Temperature and salinity.
    values = [name == :S ? salinity : Array(interior(tracers[name])) for name in tracer_names]
    fill_from_old_ocean!(values, ocean, refill_cell; maximum_passes)

    for (n, name) in enumerate(tracer_names)
        set!(tracers[name], values[n])
    end

    @info "Filled new ocean cells after pickup" number_of_new_cells number_of_deepened_columns

    # Sea-surface height, as a single level of size (Nx, Ny, 1).
    # A column is new if it is ocean now but none of its cells were ocean in the checkpoint.
    ocean_column = any(ocean, dims=3)
    new_column = ocean_column .& .!any(ocean .& .!new_cell, dims=3)
    number_of_new_columns = count(new_column)

    number_of_new_columns == 0 && return nothing

    η = ocean_model.free_surface.displacement
    η_values = [Array(interior(η))]
    fill_from_old_ocean!(η_values, ocean_column, new_column; maximum_passes)
    set!(η, η_values[1])

    @info "Filled sea-surface height in new ocean columns after pickup" number_of_new_columns

    return nothing
end

"""
    fill_from_old_ocean!(values, ocean, new_cell; maximum_passes=1000)

Fill the `new_cell` entries of every array in `values`, working outward from the old
ocean (ocean cells that are not new). Each pass, a new cell that touches at least one
ocean cell with a value (east, west, north, south, above, below) takes their average.
Cells that are never reached take the average of the old ocean at their depth level.
All arrays have size (Nx, Ny, Nz); use Nz = 1 for a single level such as η.

Returns the number of passes and the number of cells that needed the level average.
"""
function fill_from_old_ocean!(values, ocean, new_cell; maximum_passes=1000)
    Nx_grid, Ny_grid, Nz_grid = size(ocean)
    old_ocean = ocean .& .!new_cell
    needs_value = copy(new_cell)

    neighbor_offsets = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    pending = findall(new_cell)
    sums = zeros(length(values))
    passes = 0

    while !isempty(pending) && passes < maximum_passes
        passes += 1
        filled_cells = CartesianIndex{3}[]
        filled_values = Vector{Float64}[]
        still_pending = CartesianIndex{3}[]

        for cell in pending
            i, j, k = Tuple(cell)
            fill!(sums, 0)
            neighbors_used = 0

            for (di, dj, dk) in neighbor_offsets
                # The grid wraps around in longitude, but not in latitude or depth.
                ni = mod1(i + di, Nx_grid)
                nj = j + dj
                nk = k + dk
                (1 <= nj <= Ny_grid && 1 <= nk <= Nz_grid) || continue
                (ocean[ni, nj, nk] && !needs_value[ni, nj, nk]) || continue

                for n in eachindex(values)
                    sums[n] += values[n][ni, nj, nk]
                end
                neighbors_used += 1
            end

            if neighbors_used > 0
                push!(filled_cells, cell)
                push!(filled_values, sums ./ neighbors_used)
            else
                push!(still_pending, cell)
            end
        end

        # Only mark cells as filled after the pass, so every cell in a pass
        # sees the same neighbours no matter what order we visit them in.
        for (cell, cell_values) in zip(filled_cells, filled_values)
            for n in eachindex(values)
                values[n][cell] = cell_values[n]
            end
            needs_value[cell] = false
        end

        # Stop early if a pass fills nothing, since later passes would not either.
        isempty(filled_cells) && break
        pending = still_pending
    end

    # Cells with no path to the old ocean get the average of the old ocean at their level.
    number_of_level_average_cells = length(pending)
    level_averages = Dict{Tuple{Int, Int}, Float64}()
    for cell in pending
        k = cell[3]
        for n in eachindex(values)
            values[n][cell] = get!(level_averages, (n, k)) do
                level_ocean = view(old_ocean, :, :, k)
                sum(view(values[n], :, :, k)[level_ocean]) / max(1, count(level_ocean))
            end
        end
    end

    return passes, number_of_level_average_cells
end

"""
    restore_checkpoint!(simulation, filepath; check_bottom_height=true)

Restore `simulation` from the checkpoint file at `filepath`.

Oceananigans stops with an error if the sea floor saved in the checkpoint is different
from the sea floor of the current grid. Set `check_bottom_height=false` to restore anyway.
Cells that are ocean now but were sea floor in the checkpoint then come back with zero
temperature and salinity, so fill them with `fill_new_ocean_cells!` before running.
"""
function restore_checkpoint!(simulation, filepath; check_bottom_height=true)
    if check_bottom_height
        set!(simulation; checkpoint=filepath)
        return nothing
    end

    @info "Picking up simulation from checkpoint file $(abspath(filepath)) without checking the bottom height"
    state = without_saved_bottom_height(load_checkpoint_state(filepath))
    Oceananigans.restore_prognostic_state!(simulation, state)
    return nothing
end

# Oceananigans only checks the bottom height when the checkpoint has one saved,
# so removing every saved `bottom_height` skips the check.
without_saved_bottom_height(state) = state

function without_saved_bottom_height(state::NamedTuple)
    names = filter(name -> name != :bottom_height, keys(state))
    values = map(name -> without_saved_bottom_height(state[name]), names)
    return NamedTuple{names}(values)
end

function reconcile_pickup!(simulation)
    # New prescribed components are not restored from older checkpoints.
    # Put their clocks and cached input data at the restored model time.
    synchronize_pickup_component_clocks!(simulation.model)

    # The sea floor may have changed since the checkpoint was saved. Give any cells that
    # are ocean now, but were not in the checkpoint, sensible temperature and salinity,
    # and give completely new ocean columns a sea-surface height that matches their neighbours.
    # This must come before reconcile_state!, which works out the z-star stretching from η.
    ocean_model = simulation.model.ocean.model
    fill_new_ocean_cells!(ocean_model)

    # Reconcile ocean-only state such as halos, free surface, and z-star scaling.
    Oceananigans.TimeSteppers.reconcile_state!(ocean_model)

    # Rebuild coupled exchange fields and fluxes for the current component set.
    # This matters when the checkpoint was made without land but the run now has land.
    Oceananigans.TimeSteppers.update_state!(simulation.model)

    reset_catke_pickup_memory!(ocean_model.closure_fields, ocean_model.closure, ocean_model)
    compute_closure_fields!(ocean_model.closure_fields, ocean_model.closure, ocean_model)
    return nothing
end

"""
    run_segment!(state; pickup=false, reconcile_pickup=:auto, Δt=nothing,
                 stop_time=nothing, stop_iteration=nothing, wall_time_limit=nothing)

Run the simulation in `state`, picking up from a checkpoint if `pickup` is set.

With `reconcile_pickup=:auto`, the checkpoint is compared with the current grid first.
If it was saved on the same sea floor, it is restored exactly as saved, so the restart
is dynamically consistent. If the sea floor has changed, the pickup is reconciled: new
ocean cells get filled in and the free surface is rebuilt (see `reconcile_pickup!`).
Pass `true` or `false` to force one or the other.
"""
function run_segment!(state;
                      pickup=false,
                      reconcile_pickup=:auto,
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

        if reconcile_pickup === :auto
            reconcile_pickup = !checkpoint_matches_grid(simulation.model.ocean.model.grid, resolved_pickup)
            if reconcile_pickup
                @info "The checkpoint was saved on a different sea floor, so the pickup will be reconciled"
            else
                @info "The checkpoint matches the current sea floor, so it is restored exactly as saved"
            end
        end

        # A reconciled pickup fills cells that are ocean now but were not in the checkpoint,
        # so it skips the error for a changed bottom height
        restore_checkpoint!(simulation, resolved_pickup; check_bottom_height=!reconcile_pickup)
        reconcile_pickup && reconcile_pickup!(simulation)

        resolved_pickup = false
    end

    @info "Running simulation" state.run_id pickup=resolved_pickup stop_time=prettytime(simulation.stop_time)
    run!(simulation, pickup=resolved_pickup, checkpoint_at_end=true)

    return nothing
end

# To run
# state = build_simulation(GPU(), 1; add_outputs=true, Δt=5minutes)
# run_segment!(state; pickup=false, Δt=5minutes, stop_time = 1days)

# To rerun from the latest checkpoint. The pickup is reconciled only if the
# checkpoint was saved on a different sea floor (see run_segment!).
# state = nothing
# reclaim_gpu_memory!()
# state = build_simulation(GPU(), 1; add_outputs=true)
# run_segment!(state; pickup=true, Δt=10minutes, stop_time = 10000days)
