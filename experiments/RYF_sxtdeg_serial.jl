using NumericalEarth

using NumericalEarth.EN4
using NumericalEarth.ECCO
using NumericalEarth.EN4: download_dataset
using NumericalEarth.DataWrangling.ETOPO

using ClimaSeaIce
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Operators: Ax, Ay, Az, Δz,
                              Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, Δz⁻¹ᶜᶜᶠ
using Oceananigans.Fields: ReducedField, interior
using Oceananigans.Architectures: on_architecture

using OceanEnsembles

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
const output_depths = [0, -100, -500, -1000, -2000]

const checkpoint_interval = TimeInterval((365 / 48)days)
const output_interval = AveragedTimeInterval((365 / 48)days)

function get_arg(flag::String, default::Union{Nothing,String}=nothing)
    i = findfirst(==(flag), ARGS)
    if i === nothing
        return default
    elseif i == length(ARGS)
        error("Missing value for $flag")
    else
        return ARGS[i + 1]
    end
end

function parse_architecture()
    arch_str = get_arg("--arch")
    if arch_str === nothing
        println("No architecture provided. Please enter architecture (CPU/GPU):")
        arch_str = readline()
    end

    if arch_str == "GPU"
        return GPU()
    elseif arch_str == "CPU"
        return CPU()
    else
        error("Invalid architecture. Must be 'CPU' or 'GPU'.")
    end
end

function parse_run_id()
    run_str = get_arg("--run")
    if run_str === nothing
        println("No run number provided. Please enter run number:")
        run_str = readline()
    end

    return parse(Int, run_str)
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

function clear_previous_repl_state!()
    heavy_names = (:simulation,
                   :coupled_model,
                   # :sea_ice,
                   :ocean,
                   :grid,
                   :underlying_grid,
                   :bottom_height,
                   :atmosphere,
                   :radiation,
                   :forcing,
                   :FS,
                   :free_surface,
                   :closure,
                   :catke_closure,
                   :tracers,
                   :velocities,
                   :outputs,
                   :surface_height,
                   :surface_forcing,
                   :global_outputs,
                   :tot_integral_outputs,
                   :vert_integral_outputs,
                   :surf_integral_outputs,
                   :tot_integral_volumes,
                   :vert_integral_volumes,
                   :V_ccc,
                   :V_fcc,
                   :V_cfc,
                   :totint_vol_c,
                   :totint_vol_x,
                   :totint_vol_y,
                   :vertint_vol_c,
                   :vertint_vol_x,
                   :vertint_vol_y,
                   :cumulative_tuple,
                   :cumulative_vert_tuple,
                   :cumulative_tuple_vol,
                   :cumulative_vert_tuple_vol,
                   :final_state)

    for name in heavy_names
        if isdefined(Main, name)
            @eval Main $(name) = nothing
        end
    end

    reclaim_gpu_memory!()
    return nothing
end

function ryf_dates()
    return vcat(collect(DateTime(1991, 1, 1):Month(1):DateTime(1991, 4, 1)),
                collect(DateTime(1990, 5, 1):Month(1):DateTime(1990, 12, 1)))
end

function download_input_data!(dates, dataset)
    @info "Downloading/checking input data"
    @info "We download the 1990-1991 data for an RYF implementation"

    temperature = Metadata(:temperature; dates, dataset=dataset, dir=data_path)
    salinity = Metadata(:salinity; dates, dataset=dataset, dir=data_path)

    download_dataset(temperature)
    download_dataset(salinity)

    ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir=data_path)
    NumericalEarth.DataWrangling.download_dataset(ETOPOmetadata)

    return (; temperature, salinity, ETOPOmetadata)
end

function build_grid(arch, ETOPOmetadata)
    @info "Defining vertical z faces"
    z_faces = ExponentialDiscretization(Nz, depth, 0, mutable=true)
    z_surf = z_faces.cᵃᵃᶠ(Nz)

    @info "Top grid cell is " * string(abs(round(z_surf))) * "m thick"
    @info "Grid dimensions: Nx = $Nx, Ny = $Ny, Nz = $Nz"

    @info "Defining tripolar grid"
    underlying_grid = TripolarGrid(arch;
                                   size=(Nx, Ny, Nz),
                                   z=z_faces,
                                   halo=(7, 7, 7))

    @info "Defining bottom bathymetry"
    @time bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
                                            minimum_depth=15,
                                            interpolation_passes=25,
                                            major_basins=4)

    @time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

    return (; grid, z_surf)
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

function findmax_interior_field(field)
    return findmax(interior(field))
end

if isdefined(Main, :CUDA)
    @eval function findmax_interior_field(field)
        interior_field = interior(field)
        if parent(interior_field) isa CUDA.CuArray
            return findmax(Array(interior_field))
        else
            return findmax(interior_field)
        end
    end
end

function add_progress_callback!(simulation)
    wall_time = Ref(time_ns())
    callback_interval = IterationInterval(10)

    ocean_model = simulation.model.ocean.model
    grid = ocean_model.grid
    u, v, w = ocean_model.velocities

    inverse_timescale_operation =
        KernelFunctionOperation{Center, Center, Center}(advective_inverse_timescaleᶜᶜᶜ,
                                                        grid, u, v, w)

    inverse_timescale_field = Field(inverse_timescale_operation)
    longitudes = λnodes(grid, Center(), Center(), Center(); with_halos=false)
    latitudes = φnodes(grid, Center(), Center(), Center(); with_halos=false)
    depths = znodes(grid, Center(), Center(), Center(); with_halos=false)

    function progress(sim)
        η = sim.model.ocean.model.free_surface.displacement
        u, v, w = sim.model.ocean.model.velocities
        T, S = sim.model.ocean.model.tracers

        compute!(inverse_timescale_field)
        maximum_inverse_timescale, cfl_index = findmax_interior_field(inverse_timescale_field)
        i, j, k = Tuple(cfl_index)

        inverse_timescale_u, inverse_timescale_v, inverse_timescale_w =
            maybe_allow_scalar() do
                advective_inverse_timescale_componentsᶜᶜᶜ(i, j, k, grid, u, v, w)
            end

        cfl_u = sim.Δt * inverse_timescale_u
        cfl_v = sim.Δt * inverse_timescale_v
        cfl_w = sim.Δt * inverse_timescale_w
        advective_cfl = AdvectiveCFL(sim.Δt)(sim.model.ocean.model)
        dominant_component = dominant_cfl_component(cfl_u, cfl_v, cfl_w)

        cfl_longitude, cfl_latitude, cfl_depth =
            maybe_allow_scalar() do
                (longitude_value(longitudes, i, j),
                 latitude_value(latitudes, i, j),
                 depth_value(depths, i, j, k))
            end

        # ice_thickness, ice_concentration, ice_ocean_volume_flux,
        # ocean_surface_salinity, ice_salinity,
        # ice_ocean_salt_flux, ice_ocean_interface_heat_flux, ice_ocean_frazil_heat_flux,
        # ice_ocean_x_momentum_flux, ice_ocean_y_momentum_flux,
        # net_ocean_heat_flux, net_ocean_salt_flux,
        # net_ocean_x_momentum_flux, net_ocean_y_momentum_flux =
        #     maybe_allow_scalar() do
        #         sea_ice_model = sim.model.sea_ice.model
        #         ocean_model = sim.model.ocean.model
        #         sea_ice_ocean_fluxes = sim.model.interfaces.sea_ice_ocean_interface.fluxes
        #         net_ocean_fluxes = sim.model.interfaces.net_fluxes.ocean
        #         Nz = size(grid, 3)
        #         S_ocean_surface = ocean_model.tracers.S[i, j, Nz]
        #         S_ice = sea_ice_model.tracers.S[i, j, 1]
        #         J_salt = sea_ice_ocean_fluxes.salt[i, j, 1]
        #         salt_difference = S_ocean_surface - S_ice
        #         volume_flux = ifelse(abs(salt_difference) > eps(eltype(grid)),
        #                              J_salt / salt_difference,
        #                              zero(eltype(grid)))
        #
        #         (sea_ice_model.ice_thickness[i, j, 1],
        #          sea_ice_model.ice_concentration[i, j, 1],
        #          volume_flux,
        #          S_ocean_surface,
        #          S_ice,
        #          J_salt,
        #          sea_ice_ocean_fluxes.interface_heat[i, j, 1],
        #          sea_ice_ocean_fluxes.frazil_heat[i, j, 1],
        #          sea_ice_ocean_fluxes.x_momentum[i, j, 1],
        #          sea_ice_ocean_fluxes.y_momentum[i, j, 1],
        #          net_ocean_fluxes.T[i, j, 1],
        #          net_ocean_fluxes.S[i, j, 1],
        #          net_ocean_fluxes.u[i, j, 1],
        #          net_ocean_fluxes.v[i, j, 1])
        #     end

        Trange = (maximum(T), minimum(T))
        Srange = (maximum(S), minimum(S))
        ηrange = (maximum(η), minimum(η))

        umax = (maximum(abs, u),
                maximum(abs, v),
                maximum(abs, w))

        step_time = 1e-9 * (time_ns() - wall_time[])
        wall_progress = time_ns() * 1e-9

        msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
        msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
        msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Trange...)
        msg4 = @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Srange...)
        msg5 = @sprintf("extrema(η): (%.2f, %.2f) m, ", ηrange...)
        msg6 = @sprintf("wall time: %s\n", prettytime(step_time))
        msg7 = @sprintf("Wall clock time: %s\n", prettytime(wall_progress))
        msg8 = @sprintf("SYPD: %.2f\n", (10 * sim.Δt) / step_time / 365)
        msg9 = @sprintf("advective_cfl: %.2f at lat=%.2f, lon=%.2f, z=%.1f m, dominant=%s\n",
                        advective_cfl, cfl_latitude, cfl_longitude, cfl_depth, dominant_component)
        # msg10 = @sprintf("sea ice at advective_cfl max: h=%.3e m, ℵ=%.3e, qio=%.3e m s⁻¹, Ssurf=%.3e g kg⁻¹, Sice=%.3e g kg⁻¹, Jˢio=%.3e, Qio=%.3e W m⁻², Qfrazil=%.3e W m⁻², τio=(%.3e, %.3e) N m⁻², net_ocean_fluxes=(Jᵀ=%.3e m s⁻¹ ᵒC, Jˢ=%.3e m s⁻¹ g kg⁻¹, τ=(%.3e, %.3e) m² s⁻²)\n",
        #                  ice_thickness, ice_concentration, ice_ocean_volume_flux,
        #                  ocean_surface_salinity, ice_salinity,
        #                  ice_ocean_salt_flux, ice_ocean_interface_heat_flux, ice_ocean_frazil_heat_flux,
        #                  ice_ocean_x_momentum_flux, ice_ocean_y_momentum_flux,
        #                  net_ocean_heat_flux, net_ocean_salt_flux,
        #                  net_ocean_x_momentum_flux, net_ocean_y_momentum_flux)

        @info msg1 * msg2 * msg3 * msg4 * msg5 * msg6 * msg7 * msg8 * msg9
        # @info msg1 * msg2 * msg3 * msg4 * msg5 * msg6 * msg7 * msg8 * msg9 * msg10

        wall_time[] = time_ns()
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

function add_run_output_writers!(simulation, ocean, grid, run_id)
    run_id_leading = lpad(string(run_id), 4, '0')
    @info "Defining run-dependent output writers for run $run_id_leading"

    outputs = merge(ocean.model.tracers, ocean.model.velocities)
    surface_height = (; surface_height=ocean.model.free_surface.displacement)
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
                                                          overwrite_existing=true,
                                                          array_type=Array{Float32})
    end

    @time ocean.output_writers[:SSH] = JLD2Writer(ocean.model, surface_height;
                                                  dir=output_path,
                                                  schedule=output_interval,
                                                  filename="global_ssh_fields_sxtdeg_RYF_run" * run_id_leading,
                                                  with_halos=false,
                                                  overwrite_existing=true,
                                                  array_type=Array{Float32})

    @time simulation.output_writers[:surface_fluxes] = JLD2Writer(simulation.model, surface_forcing;
                                                                  dir=output_path,
                                                                  schedule=output_interval,
                                                                  filename="global_surface_fluxes_sxtdeg_RYF_run" * run_id_leading,
                                                                  with_halos=false,
                                                                  overwrite_existing=true,
                                                                  array_type=Array{Float32})

    @time ocean.output_writers[:integral] = JLD2Writer(ocean.model, build_global_outputs(ocean, grid);
                                                       dir=output_path,
                                                       schedule=output_interval,
                                                       filename="global_tot_integrals_sxtdeg_RYF_run" * run_id_leading,
                                                       overwrite_existing=true)

    return nothing
end

function build_simulation(arch, run_id; add_outputs=true)
    dates = ryf_dates()
    dataset = EN4Monthly()
    inputs = download_input_data!(dates, dataset)

    @info "Defining grid"
    grid_state = build_grid(arch, inputs.ETOPOmetadata)
    grid = grid_state.grid
    z_surf = grid_state.z_surf

    @info "Defining restoring rate"
    restoring_rate = 1 / 30days
    mask(x, y, z, t) = z >= z_surf - 1

    # Keep time caches small so pickup does not require a large transient memory budget.
    FS = DatasetRestoring(inputs.salinity, grid; mask, rate=restoring_rate, time_indices_in_memory=2)
    forcing = (; S=FS)

    @info "Defining closures"
    catke_closure = NumericalEarth.Oceans.default_ocean_closure()
    closure = (catke_closure, VerticalScalarDiffusivity(κ=1e-5, ν=1e-4))

    @info "Defining free surface"
    free_surface = SplitExplicitFreeSurface(grid; substeps=70)
    momentum_advection = WENOVectorInvariant()
    tracer_advection = WENO(order=7)

    @info "Defining ocean model"
    @time ocean = ocean_simulation(grid; Δt=1minutes,
                                   momentum_advection,
                                   tracer_advection,
                                   timestepper=:SplitRungeKutta3,
                                   free_surface,
                                   forcing,
                                   radiative_forcing=nothing,
                                   closure)

    @info "Initialising with EN4"
    set!(ocean.model,
         T=Metadata(:temperature; dates=first(dates), dataset=dataset, dir=data_path),
         S=Metadata(:salinity; dates=first(dates), dataset=dataset, dir=data_path))

    # Sea ice is disabled for this fresh ocean-atmosphere-radiation run.
    # @info "Creating sea ice model"
    # sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7))
    #
    # set!(sea_ice.model,
    #      h=Metadatum(:sea_ice_thickness; dataset=ECCO4Monthly(), dir=data_path),
    #      ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly(), dir=data_path))

    @info "Defining Atmospheric state"
    radiation = Radiation(arch)
    atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory=2, include_rivers_and_icebergs=true)

    @info "Defining coupled model"
    # @time coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
    @time coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

    simulation = Simulation(coupled_model; Δt=10minutes)
    add_progress_callback!(simulation)

    if add_outputs
        add_run_output_writers!(simulation, ocean, grid, run_id)
    end

    @time simulation.output_writers[:checkpointer] = Checkpointer(coupled_model,
                                                                  schedule=checkpoint_interval,
                                                                  dir=output_path,
                                                                  prefix="RYF_sxtdeg_checkpoint",
                                                                  overwrite_existing=true,
                                                                  cleanup=false)

    return (; simulation, ocean)
end

function run_segment!(state, run_id; pickup)
    simulation = state.simulation
    simulation.Δt = 10minutes
    simulation.stop_time = run_id * 11 * (365 / 12)days

    @info "Running simulation" run_id pickup stop_time=prettytime(simulation.stop_time)
    run!(simulation, pickup=pickup, checkpoint_at_end=true)

    return nothing
end

function run_with_restart_rebuild!(arch, run_id)
    pickup = run_id > 1
    gpu_memory_status("Before build: ")

    state = build_simulation(arch, run_id)
    run_segment!(state, run_id; pickup)

    next_run_id = run_id + 1

    @info "Dropping completed run from memory before pickup restart" completed_run=run_id next_run=next_run_id
    state = nothing
    reclaim_gpu_memory!()

    state = build_simulation(arch, next_run_id)
    run_segment!(state, next_run_id; pickup=true)

    return state
end

clear_previous_repl_state!()

arch = parse_architecture()
run_id = parse_run_id()

final_state = run_with_restart_rebuild!(arch, run_id)
