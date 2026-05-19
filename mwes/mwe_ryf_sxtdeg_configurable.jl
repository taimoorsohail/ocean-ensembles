module MWERYFSxtdegConfigurable

using NumericalEarth

using NumericalEarth.EN4
using NumericalEarth.ECCO
using NumericalEarth.DataWrangling.ETOPO

using ClimaSeaIce
using CUDA

using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: interior

using Dates
using Printf
using JLD2

const data_path = expanduser("/home/tsohail/uom/ocean-ensembles/data/")
const output_path = expanduser("/home/tsohail/uom/ocean-ensembles/outputs/")

const Nx = Integer(360 * 6)
const Ny = Integer(180 * 6)
const Nz = Integer(75)
const depth = -5500.0

const RUN_LABEL = "mwe_ryf_sxtdeg_configurable"

Base.@kwdef struct Config
    arch = GPU(CUDA.CUDABackend())
    include_atmosphere::Bool = false
    include_radiation::Bool = false
    include_sea_ice::Bool = false
    include_restoring::Bool = false
    momentum_advection = WENOVectorInvariant()
    tracer_advection = WENO(order = 7)
    initial_ts_dataset = EN4Monthly()
    initial_momentum_dataset = ECCO4Monthly()
    initial_momentum_date::DateTime = DateTime(1998, 1, 1)
    ocean_timestep = 1minutes
    coupled_timestep = 10minutes
    stop_time = 1days
    output_schedule = IterationInterval(1)
    progress_schedule = IterationInterval(10)
    tendency_record::Bool = true
    surface_level::Int = Nz
    run_label::String = RUN_LABEL
end

function ryf_dates()
    return vcat(collect(DateTime(1991, 1, 1):Month(1):DateTime(1991, 4, 1)),
                collect(DateTime(1990, 5, 1):Month(1):DateTime(1990, 12, 1)))
end

function build_grid(arch, ETOPOmetadata)
    z_faces = ExponentialDiscretization(Nz, depth, 0, mutable = true)

    underlying_grid = TripolarGrid(arch;
                                   size = (Nx, Ny, Nz),
                                   z = z_faces,
                                   halo = (7, 7, 7))

    @time bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
                                            minimum_depth = 15,
                                            interpolation_passes = 25,
                                            major_basins = 2)

    @time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map = true)

    return grid
end

function maybe_download!(metadata)
    NumericalEarth.DataWrangling.download_dataset(metadata)
    return metadata
end

function input_metadata(config::Config)
    dates = ryf_dates()

    temperature = maybe_download!(Metadata(:temperature; dates, dataset = config.initial_ts_dataset, dir = data_path))
    salinity = maybe_download!(Metadata(:salinity; dates, dataset = config.initial_ts_dataset, dir = data_path))
    bathymetry = maybe_download!(Metadatum(:bottom_height; dataset = ETOPO2022(), dir = data_path))

    momentum = if isnothing(config.initial_momentum_dataset)
        nothing
    else
        (u = maybe_download!(Metadata(:u_velocity; dates = config.initial_momentum_date, dataset = config.initial_momentum_dataset, dir = data_path)),
         v = maybe_download!(Metadata(:v_velocity; dates = config.initial_momentum_date, dataset = config.initial_momentum_dataset, dir = data_path)))
    end

    return (; dates, temperature, salinity, bathymetry, momentum)
end

function optional_salinity_restoring(config::Config, grid, dates)
    config.include_restoring || return NamedTuple()

    z_surf = grid.z.cᵃᵃᶠ(Nz)
    restoring_rate = 1 / 30days
    mask(x, y, z, t) = z >= z_surf - 1

    salinity = Metadata(:salinity; dates, dataset = config.initial_ts_dataset, dir = data_path)
    restoring = DatasetRestoring(salinity, grid; mask, rate = restoring_rate, time_indices_in_memory = 2)

    return (; S = restoring)
end

function build_ocean(config::Config, grid, inputs)
    free_surface = SplitExplicitFreeSurface(grid; substeps = 70)
    forcing = optional_salinity_restoring(config, grid, inputs.dates)

    closure = (NumericalEarth.Oceans.default_ocean_closure(),
               VerticalScalarDiffusivity(κ = 1e-5, ν = 1e-4))

    @time ocean = ocean_simulation(grid; Δt = config.ocean_timestep,
                                   momentum_advection = config.momentum_advection,
                                   tracer_advection = config.tracer_advection,
                                   timestepper = :SplitRungeKutta3,
                                   free_surface,
                                   forcing,
                                   radiative_forcing = nothing,
                                   closure)

    set!(ocean.model,
         T = Metadata(:temperature; dates = first(inputs.dates), dataset = config.initial_ts_dataset, dir = data_path),
         S = Metadata(:salinity; dates = first(inputs.dates), dataset = config.initial_ts_dataset, dir = data_path))

    if !isnothing(inputs.momentum)
        set!(ocean.model, u = inputs.momentum.u, v = inputs.momentum.v)
    end

    return ocean
end

function build_sea_ice(config::Config, grid, ocean)
    config.include_sea_ice || return nothing

    sea_ice = sea_ice_simulation(grid, ocean; advection = WENO(order = 7, minimum_buffer_upwind_order = 1))
    set!(sea_ice.model,
         h = Metadatum(:sea_ice_thickness; dataset = ECCO4Monthly(), dir = data_path),
         ℵ = Metadatum(:sea_ice_concentration; dataset = ECCO4Monthly(), dir = data_path))

    return sea_ice
end

function build_atmosphere(config::Config)
    config.include_atmosphere || return (; atmosphere = nothing, land = nothing)

    backend = JRA55NetCDFBackend(2)
    atmosphere = JRA55PrescribedAtmosphere(config.arch; backend)
    land = JRA55PrescribedLand(config.arch; backend)
    return (; atmosphere, land)
end

function build_radiation(config::Config)
    config.include_radiation || return nothing
    return JRA55PrescribedRadiation(config.arch; backend = JRA55NetCDFBackend(2))
end

function build_model(ocean, sea_ice, atmosphere, land, radiation)
    if isnothing(sea_ice) && isnothing(atmosphere) && isnothing(radiation)
        return ocean.model
    elseif isnothing(sea_ice)
        return OceanOnlyModel(ocean; atmosphere, land, radiation)
    else
        kwargs = (;)
        !isnothing(atmosphere) && (kwargs = merge(kwargs, (; atmosphere)))
        !isnothing(radiation) && (kwargs = merge(kwargs, (; radiation)))
        return OceanSeaIceModel(sea_ice, ocean; kwargs...)
    end
end

function ocean_model_from(simulation)
    model = simulation.model
    return hasproperty(model, :ocean) ? model.ocean.model : model
end

surface_array(field, k) = Array(interior(field)[:, :, k])

function add_progress_callback!(config::Config, simulation)
    wall_time = Ref(time_ns())

    function progress(sim)
        ocean_model = ocean_model_from(sim)
        u, v, w = ocean_model.velocities
        T, S = ocean_model.tracers

        umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))
        Trange = extrema(T)
        Srange = extrema(S)
        step_time = 1e-9 * (time_ns() - wall_time[])

        @info @sprintf("iter=%d time=%s Δt=%s max|u|=(%.2e, %.2e, %.2e) T=(%.2f, %.2f) S=(%.2f, %.2f) wall=%s",
                       iteration(sim), prettytime(sim), prettytime(sim.Δt),
                       umax..., Trange..., Srange..., prettytime(step_time))

        wall_time[] = time_ns()
        return nothing
    end

    add_callback!(simulation, progress, config.progress_schedule)
    return nothing
end

function add_surface_output_writer!(config::Config, ocean)
    outputs = merge(ocean.model.tracers, ocean.model.velocities)

    ocean.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                                dir = output_path,
                                                schedule = config.output_schedule,
                                                filename = config.run_label * "_surface_k$(config.surface_level)",
                                                indices = (:, :, config.surface_level),
                                                with_halos = false,
                                                overwrite_existing = true,
                                                array_type = Array{Float32})

    return nothing
end

function add_tendency_callback!(config::Config, simulation)
    config.tendency_record || return nothing

    ocean_model = ocean_model_from(simulation)
    u, v, w = ocean_model.velocities

    previous_time = Ref(simulation.model.clock.time)
    previous_u = Ref(surface_array(u, config.surface_level))
    previous_v = Ref(surface_array(v, config.surface_level))
    previous_w = Ref(surface_array(w, config.surface_level))

    function record_tendency(sim)
        ocean_model = ocean_model_from(sim)
        u, v, w = ocean_model.velocities

        current_time = sim.model.clock.time
        Δt = current_time - previous_time[]
        Δt <= 0 && return nothing

        current_u = surface_array(u, config.surface_level)
        current_v = surface_array(v, config.surface_level)
        current_w = surface_array(w, config.surface_level)

        du_dt = (current_u .- previous_u[]) ./ Δt
        dv_dt = (current_v .- previous_v[]) ./ Δt
        dw_dt = (current_w .- previous_w[]) ./ Δt

        filepath = joinpath(output_path, config.run_label * @sprintf("_tendency_iter%06d.jld2", iteration(sim)))
        jldsave(filepath;
                time = current_time,
                Δt = Δt,
                surface_level = config.surface_level,
                u = current_u,
                v = current_v,
                w = current_w,
                du_dt,
                dv_dt,
                dw_dt)

        previous_time[] = current_time
        previous_u[] = current_u
        previous_v[] = current_v
        previous_w[] = current_w

        return nothing
    end

    add_callback!(simulation, record_tendency, config.output_schedule)
    return nothing
end

function main(; config = Config())
    @info "Building configurable RYF sxtdeg MWE" config

    inputs = input_metadata(config)
    grid = build_grid(config.arch, inputs.bathymetry)
    @time ocean = build_ocean(config, grid, inputs)
    sea_ice = build_sea_ice(config, grid, ocean)
    @time atmosphere_state = build_atmosphere(config)
    @time radiation = build_radiation(config)

    model = build_model(ocean, sea_ice, atmosphere_state.atmosphere, atmosphere_state.land, radiation)
    simulation = Simulation(model; Δt = config.coupled_timestep, stop_time = config.stop_time)

    add_progress_callback!(config, simulation)
    add_surface_output_writer!(config, ocean)
    add_tendency_callback!(config, simulation)

    @info "Running configurable MWE" stop_time = prettytime(config.stop_time) output_schedule = config.output_schedule
    run!(simulation)

    return simulation
end

export Config, main

end

if abspath(PROGRAM_FILE) == @__FILE__
    using .MWERYFSxtdegConfigurable
    MWERYFSxtdegConfigurable.main()
end
