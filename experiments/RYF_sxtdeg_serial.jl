using NumericalEarth

using NumericalEarth.EN4
using NumericalEarth.ECCO
using NumericalEarth.DataWrangling: download_dataset
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
                              Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, Δz⁻¹ᶜᶜᶠ
using Oceananigans.Fields: ReducedField, interior, ConstantField, ZeroField, OneField
using Oceananigans.ImmersedBoundaries: immersed_cell, peripheral_node
using Oceananigans.Architectures: on_architecture

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

checkpoint_interval = TimeInterval((365/48)days)
output_interval = AveragedTimeInterval((365/48)days)
diagnostic_surface_interval = TimeInterval((365/48)days)
callback_iteration_interval = 10

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
    # z_surf = z_faces[Nz]
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
                                            major_basins=2)

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

function add_progress_callback!(simulation; callback_iteration_interval = callback_iteration_interval)
    start_wall_time = Ref(time_ns())
    wall_time = Ref(time_ns())
    callback_interval = IterationInterval(callback_iteration_interval)

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
        iteration = Oceananigans.iteration(sim)

        compute!(inverse_timescale_field)
        maximum_inverse_timescale, cfl_index = findmax_interior_field(inverse_timescale_field)
        i, j, k = Tuple(cfl_index)

        inverse_timescale_u, inverse_timescale_v, inverse_timescale_w =
            maybe_allow_scalar() do
                advective_inverse_timescale_componentsᶜᶜᶜ(i, j, k, grid, u, v, w)
            end

        uC, vC, wC, wA, uE, vN =
            maybe_allow_scalar() do
                (u[i, j, k],
                 v[i, j, k],
                 w[i, j, k],
                 w[i, j, k+1],
                 u[i+1, j, k],
                 v[i, j+1, k])
            end

        bottom = grid.immersed_boundary.bottom_height
        hC, hW, hE, hS, hN, local_Δz, local_Δz_above, local_cfl_w, local_cfl_w_above =
            maybe_allow_scalar() do
                Δz⁻¹ = Δz⁻¹ᶜᶜᶠ(i, j, k, grid)
                Δz⁻¹_above = Δz⁻¹ᶜᶜᶠ(i, j, k+1, grid)

                (bottom[i, j, 1],
                 bottom[i-1, j, 1],
                 bottom[i+1, j, 1],
                 bottom[i, j-1, 1],
                 bottom[i, j+1, 1],
                 1 / Δz⁻¹,
                 1 / Δz⁻¹_above,
                 sim.Δt * abs(w[i, j, k]) * Δz⁻¹,
                 sim.Δt * abs(w[i, j, k+1]) * Δz⁻¹_above)
            end

        cell_immersed, uC_peripheral, uE_peripheral, vC_peripheral, vN_peripheral =
            maybe_allow_scalar() do
                (immersed_cell(i, j, k, grid),
                 peripheral_node(i, j, k, grid, Face(), Center(), Center()),
                 peripheral_node(i+1, j, k, grid, Face(), Center(), Center()),
                 peripheral_node(i, j, k, grid, Center(), Face(), Center()),
                 peripheral_node(i, j+1, k, grid, Center(), Face(), Center()))
            end

        cell_above_immersed, wC_peripheral, wA_peripheral =
            maybe_allow_scalar() do
                (k < grid.Nz ? immersed_cell(i, j, k+1, grid) : false,
                 peripheral_node(i, j, k, grid, Center(), Center(), Face()),
                 peripheral_node(i, j, k+1, grid, Center(), Center(), Face()))
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
        msg9 = @sprintf("advective_cfl: %.2f at lat=%.2f, lon=%.2f, z=%.1f m, dominant=%s\n",
                        advective_cfl, cfl_latitude, cfl_longitude, cfl_depth, dominant_component)
        msg10 = @sprintf("cfl components: u/v/w=(%.2f, %.2f, %.2f)\n",
                         cfl_u, cfl_v, cfl_w)
        msg11 = @sprintf("cfl-site velocities: uC/uE=(%.2e, %.2e), vC/vN=(%.2e, %.2e), wC/wA=(%.2e, %.2e)\n",
                         uC, uE, vC, vN, wC, wA)
        msg12 = @sprintf("cfl-site mask: cell_immersed=%s, uC/uE peripheral=(%s, %s), vC/vN peripheral=(%s, %s)\n",
                         string(cell_immersed), string(uC_peripheral), string(uE_peripheral),
                         string(vC_peripheral), string(vN_peripheral))
        msg13 = @sprintf("cfl-site vertical mask: cell_above_immersed=%s, wC/wA peripheral=(%s, %s)\n",
                         string(cell_above_immersed), string(wC_peripheral), string(wA_peripheral))
        msg14 = @sprintf("cfl-site bottom hC/hW/hE/hS/hN=(%.1f, %.1f, %.1f, %.1f, %.1f) m\n",
                         hC, hW, hE, hS, hN)
        msg15 = @sprintf("cfl-site Δz/Δz_above=(%.2f, %.2f) m, cfl_w/cfl_w_above=(%.2f, %.2f)\n",
                         local_Δz, local_Δz_above, local_cfl_w, local_cfl_w_above)

        @info msg1 * msg2 * msg3 * msg4 * msg5 * msg6 * msg7 * msg8 * msg9 * msg10 * msg11 * msg12 * msg13 * msg14 * msg15

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

function filter_diagnostic_surface_outputs(outputs)
    names = Symbol[]
    fields = []

    for name in keys(outputs)
        output = outputs[name]

        if output isa DiagnosticConstantField
            @info "Skipping constant diagnostic surface output" name output
        else
            push!(names, name)
            push!(fields, output)
        end
    end

    return NamedTuple{Tuple(names)}(Tuple(fields))
end

function build_diagnostic_surface_outputs(simulation)
    ocean_model = simulation.model.ocean.model
    sea_ice_model = simulation.model.sea_ice.model

    sea_ice_ocean_fluxes = simulation.model.interfaces.sea_ice_ocean_interface.fluxes
    atmosphere_ocean_fluxes = simulation.model.interfaces.atmosphere_ocean_interface.fluxes
    net_ocean_fluxes = simulation.model.interfaces.net_fluxes.ocean

    base_outputs = (;
        surface_height = ocean_model.free_surface.displacement,
        net_ocean_flux_T = net_ocean_fluxes.T,
        net_ocean_flux_S = net_ocean_fluxes.S,
        net_ocean_flux_u = net_ocean_fluxes.u,
        net_ocean_flux_v = net_ocean_fluxes.v,
        atmosphere_ocean_sensible_heat = atmosphere_ocean_fluxes.sensible_heat,
        atmosphere_ocean_latent_heat = atmosphere_ocean_fluxes.latent_heat,
        atmosphere_ocean_water_vapor = atmosphere_ocean_fluxes.water_vapor,
        atmosphere_ocean_x_momentum = atmosphere_ocean_fluxes.x_momentum,
        atmosphere_ocean_y_momentum = atmosphere_ocean_fluxes.y_momentum,
        sea_ice_ocean_interface_heat = sea_ice_ocean_fluxes.interface_heat,
        sea_ice_ocean_frazil_heat = sea_ice_ocean_fluxes.frazil_heat,
        sea_ice_ocean_salt = sea_ice_ocean_fluxes.salt,
        sea_ice_ocean_x_momentum = sea_ice_ocean_fluxes.x_momentum,
        sea_ice_ocean_y_momentum = sea_ice_ocean_fluxes.y_momentum,
        sea_ice_thickness = sea_ice_model.ice_thickness,
        sea_ice_consolidation_thickness = sea_ice_model.ice_consolidation_thickness,
        sea_ice_concentration = sea_ice_model.ice_concentration,
        sea_ice_salinity = sea_ice_model.tracers.S,
        sea_ice_top_surface_temperature = sea_ice_model.ice_thermodynamics.top_surface_temperature,
        sea_ice_u = sea_ice_model.velocities.u,
        sea_ice_v = sea_ice_model.velocities.v)

    radiation = simulation.model.radiation
    radiation_interface_fluxes = isnothing(radiation) ? nothing : radiation.interface_fluxes
    ocean_radiation_fluxes = if isnothing(radiation_interface_fluxes) || !haskey(radiation_interface_fluxes, :ocean)
        nothing
    else
        radiation_interface_fluxes.ocean
    end

    radiation_outputs = isnothing(ocean_radiation_fluxes) ? NamedTuple() : (;
        ocean_radiation_upwelling_longwave = ocean_radiation_fluxes.upwelling_longwave,
        ocean_radiation_downwelling_longwave = ocean_radiation_fluxes.downwelling_longwave,
        ocean_radiation_downwelling_shortwave = ocean_radiation_fluxes.downwelling_shortwave)

    return filter_diagnostic_surface_outputs(merge(base_outputs, radiation_outputs))
end

function remove_existing_diagnostic_output_files!(run_id_leading)
    diagnostic_filenames = (
        "global_diagnostic_k$(Nz - 1)_fields_sxtdeg_RYF_run" * run_id_leading,
        "global_diagnostic_surface_fields_sxtdeg_RYF_run" * run_id_leading)

    for filename in diagnostic_filenames
        filepath = joinpath(output_path, filename * ".jld2")
        if isfile(filepath)
            @info "Removing existing diagnostic output before pickup/restart" filepath
            rm(filepath; force=true)
        end
    end

    return nothing
end

function add_run_output_writers!(simulation, ocean, grid, run_id)
    run_id_leading = lpad(string(run_id), 4, '0')
    @info "Defining run-dependent output writers for run $run_id_leading"

    outputs = merge(ocean.model.tracers, ocean.model.velocities)
    diagnostic_subsurface_outputs = merge(outputs, (; e=ocean.model.tracers.e))
    diagnostic_surface_level = Nz - 1
    remove_existing_diagnostic_output_files!(run_id_leading)
    surface_height = (; surface_height=ocean.model.free_surface.displacement)
    # Surface flux diagnostics are disabled for the ocean-only run because these
    # helpers assume a sea-ice-ocean interface exists in this NumericalEarth version.
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

    # @time ocean.output_writers[:diagnostic_subsurface] = JLD2Writer(ocean.model, diagnostic_subsurface_outputs;
    #                                                                 dir=output_path,
    #                                                                 schedule=diagnostic_surface_interval,
    #                                                                 filename="global_diagnostic_k$(diagnostic_surface_level)_fields_sxtdeg_RYF_run" * run_id_leading,
    #                                                                 indices=(:, :, diagnostic_surface_level),
    #                                                                 including=(),
    #                                                                 with_halos=false,
    #                                                                 overwrite_existing=true,
    #                                                                 array_type=Array{Float32})

    # @time simulation.output_writers[:diagnostic_surface] = JLD2Writer(simulation.model, build_diagnostic_surface_outputs(simulation);
    #                                                                   dir=output_path,
    #                                                                   schedule=diagnostic_surface_interval,
    #                                                                   filename="global_diagnostic_surface_fields_sxtdeg_RYF_run" * run_id_leading,
    #                                                                   including=(),
    #                                                                   with_halos=false,
    #                                                                   overwrite_existing=true,
    #                                                                   array_type=Array{Float32})

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

    @info "Defining free surface" #Try running w/o sea ice duna,mics, remove rivers and iceberges? 
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

    @info "Creating sea ice model"
    # sea_ice_rheology = ElastoViscoPlasticRheology(rheology_activation_concentration = (0.15, 0.80))
    # sea_ice_dynamics = NumericalEarth.SeaIces.sea_ice_dynamics(grid, ocean; rheology = sea_ice_rheology)
    sea_ice = sea_ice_simulation(grid, ocean;
                                 advection = WENO(order=7,
                                 minimum_buffer_upwind_order=1))

    set!(sea_ice.model,
         h=Metadatum(:sea_ice_thickness; dataset=ECCO4Monthly(), dir=data_path),
         ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly(), dir=data_path))

    @info "Defining Atmospheric state"
    time_indices_in_memory = 24
    radiation = JRA55PrescribedRadiation(arch; time_indices_in_memory)
    atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory)
    land = JRA55PrescribedLand(arch; time_indices_in_memory)

    @info "Defining coupled model"
    interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
    radiation,
    sea_ice_ocean_salinity_flux = nothing)

    @time coupled_model = OceanSeaIceModel(sea_ice, ocean;
        atmosphere,
        radiation,
        interfaces)
    # @time coupled_model = OceanSeaIceModel(sea_ice, ocean; atmosphere, radiation)
    # @time coupled_model = OceanOnlyModel(ocean; atmosphere, land, radiation)

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

    return (; simulation, ocean, run_id)
end

function run_segment!(state; pickup=false, Δt=5minutes, stop_time = nothing, stop_iteration = nothing)
    simulation = state.simulation
    simulation.Δt = Δt
    if isnothing(stop_time) && isnothing(stop_iteration)
        simulation.stop_time = state.run_id * 12 * (365 / 12)days
    elseif !isnothing(stop_iteration) && isnothing(stop_time)
        simulation.stop_iteration = stop_iteration
    elseif isnothing(stop_iteration) && !isnothing(stop_time)
        simulation.stop_time = stop_time
    elseif !isnothing(stop_iteration) && !isnothing(stop_time)
        error("Only one of stop_time or stop_iteration should be provided")
    end

    @info "Running simulation" state.run_id pickup stop_time=prettytime(simulation.stop_time)
    run!(simulation, pickup=pickup, checkpoint_at_end=true)

    return nothing
end

# To run
# state = build_simulation(GPU(), 1; add_outputs=true)
# run_segment!(state; pickup=false, Δt=10minutes, stop_time = 1days
