using NumericalEarth
using NumericalEarth.DataWrangling: SurfaceFluxRestoring,
                                    ConservativeSurfaceFluxRestoring,
                                    ConservativeSurfaceFluxRestoringCallback

using NumericalEarth.Oceans: river_mouth_vertical_diffusivity
using NumericalEarth.Diagnostics: MixedLayerDepthField, net_ocean_heat_flux, net_ocean_freshwater_flux
using NumericalEarth.DataWrangling.SeaWiFS: SeaWiFSMonthly
using NumericalEarth.EN4
using NumericalEarth.ECCO
using NumericalEarth.DataWrangling.ETOPO
using NumericalEarth.EarthSystemModels.InterfaceComputations: IceBathHeatFlux, ThreeEquationHeatFlux, MomentumBasedFrictionVelocity,
                                                              ZeroHeatContentMeltwater
# Pieces of Simone's OMIP air-sea and air-ice flux formulas (added 2026-11-26)
using NumericalEarth.EarthSystemModels.InterfaceComputations: COARELogarithmicSimilarityProfile,
                                                              WindDependentWaveFormulation,
                                                              MomentumRoughnessLength,
                                                              TemperatureDependentAirViscosity,
                                                              ScalarRoughnessLength,
                                                              SimilarityTheoryFluxes,
                                                              atmosphere_sea_ice_stability_functions,
                                                              ConvectiveGustiness,
                                                              AtmosphericThermodynamics,
                                                              ImpureSaturationSpecificHumidity,
                                                              GillSaturationEnhancement
using ClimaSeaIce.SeaIceThermodynamics: ConductiveFlux, UniformThicknessDistribution, PrescribedTemperature,
                                        snow_slab_thermodynamics

# Ice-ocean heat flux copied unchanged from Simone's OMIP setup (NumericalEarth ss/omip-prototype,
# experiments/OMIPSimulations/src/omip_simulation.jl). The default (fixed u★ = 0.002) gives a melt rate q
# in open water too, which added a heat sink of about 7 W/m² to our run (claude-notes, 2026-11-24).
"""
    corrected_ice_ocean_heat_flux()

Three-equation ice-ocean heat flux with momentum-based friction velocity
computed from actual ice-ocean stress (McPhee 1992, 2008; SHEBA median u*≈0.01 m/s).

`heat_transfer_coefficient` defaults to McPhee's Stanton number 0.0057, which is the value
consistent with a *computed* friction velocity. The 0.0095 of Shi et al. (2021) is calibrated
against a fixed u* = 0.002 m/s; pairing it with u* = sqrt(|tau|/rho), which is ~0.006 m/s under the
Arctic pack, inflates the exchange velocity alpha_h u* by a factor of three.
The salt transfer coefficient follows at the standard ratio R = alpha_h / alpha_s = 35.
"""
corrected_ice_ocean_heat_flux(; heat_transfer_coefficient = 0.0057) =
    ThreeEquationHeatFlux(; heat_transfer_coefficient,
                            salt_transfer_coefficient = heat_transfer_coefficient / 35,
                            friction_velocity = MomentumBasedFrictionVelocity())

# The three functions below are copied unchanged from Simone's OMIP setup (NumericalEarth ss/omip-prototype
# c5ad24b9, experiments/OMIPSimulations/src/omip_simulation.jl), the formulas his runs use by default
# (OCEAN_FLUXES=corrected, ICE_FLUXES=corrected). Added 2026-11-26 against the warming drift. Compared with the
# NumericalEarth defaults they add a minimum gust of 0.5 m/s over the ocean, a sea-surface roughness that grows
# with wind speed, a temperature-dependent air viscosity, the COARE form of the wind and temperature profiles,
# and fixed roughness lengths over sea ice.
function corrected_atmosphere_ocean_fluxes(FT = Float64;
                                           subgrid_velocities = ConvectiveGustiness{FT}(minimum_gustiness = FT(0.5)))
    air_kinematic_viscosity = TemperatureDependentAirViscosity(FT)
    return SimilarityTheoryFluxes(FT;
                                  similarity_form              = COARELogarithmicSimilarityProfile(),
                                  subgrid_velocities           = subgrid_velocities,
                                  momentum_roughness_length    = MomentumRoughnessLength(FT;
                                  wave_formulation             = WindDependentWaveFormulation(FT),
                                  air_kinematic_viscosity      = TemperatureDependentAirViscosity(FT)),
                                  temperature_roughness_length = ScalarRoughnessLength(FT; air_kinematic_viscosity),
                                  water_vapor_roughness_length = ScalarRoughnessLength(FT; air_kinematic_viscosity))
end

"""
    corrected_atmosphere_sea_ice_fluxes(FT = Float64; momentum_roughness_length, scalar_roughness_length)

Atmosphere-sea ice flux formulation with:
- SHEBA/Paulson+Grachev stability functions (existing default, correct)
- Fixed momentum roughness z0 = 5e-4 m (CICE/SHEBA standard; Andreas et al. 2010)
- Fixed scalar roughness z0t = z0q = 5e-5 m (Andreas 1987: z0t ≈ z0/10 at R*≈7)
- COARE logarithmic similarity profile
- Minimum gustiness = 0.2 m/s

Both roughnesses are geometric constants rather than wind-dependent, because sea ice carries no gravity
waves; the roughness is set by ridges, floe edges and sastrugi. The SHEBA value describes multiyear pack,
and smoother first-year ice sits nearer 1e-4 m, so `momentum_roughness_length` is exposed to let the drift
speed be varied over its observed range.
"""
corrected_atmosphere_sea_ice_fluxes(FT = Float64;
                                    momentum_roughness_length = 5e-4,
                                    scalar_roughness_length = 5e-5) =
    SimilarityTheoryFluxes(FT;
                           stability_functions          = atmosphere_sea_ice_stability_functions(FT),
                           similarity_form              = COARELogarithmicSimilarityProfile(),
                           subgrid_velocities           = ConvectiveGustiness{FT}(minimum_gustiness = FT(0.2)),
                           momentum_roughness_length    = FT(momentum_roughness_length),
                           temperature_roughness_length = FT(scalar_roughness_length),
                           water_vapor_roughness_length = FT(scalar_roughness_length))

# The saturation vapor pressure over both the ocean and the sea ice carries the Gill (1982) moist-air enhancement
function saturation_enhanced_humidities(FT)
    saturation_enhancement = GillSaturationEnhancement(FT)
    ocean = ImpureSaturationSpecificHumidity(AtmosphericThermodynamics.Liquid(), convert(FT, 0.98); saturation_enhancement)
    sea_ice = ImpureSaturationSpecificHumidity(AtmosphericThermodynamics.Ice(); saturation_enhancement)
    return (; ocean, sea_ice)
end


using ClimaSeaIce
using ClimaSeaIce.Rheologies: ElastoViscoPlasticRheology
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Oceananigans.Operators: Ax, Ay, Az,
                              Δx⁻¹ᶠᶜᶜ, Δy⁻¹ᶜᶠᶜ, Δz⁻¹ᶜᶜᶠ, div_xyᶜᶜᶜ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.Fields: ReducedField, interior, ConstantField, ZeroField, OneField, AbstractField
using Oceananigans.ImmersedBoundaries: immersed_cell, peripheral_node
using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: FaceFolded, λnode, φnode, znode, φnodes
using Oceananigans.TimeSteppers: VerticallyImplicitTimeDiscretization, AdaptiveVerticallyImplicitDiscretization
using Oceananigans.Utils: ExplicitTimeDiscretization, KernelParameters
using Oceananigans.Grids: static_column_depthᶜᶜᵃ
using Oceananigans.Models.HydrostaticFreeSurfaceModels: update_zstar_scaling!
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: architecture
using Oceananigans.Utils.KernelAbstractions: @kernel, @index
using Oceananigans.TurbulenceClosures: compute_closure_fields!
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEMixingLength, CATKEEquation
using Oceananigans.OutputWriters: load_checkpoint_state

using SeawaterPolynomials.TEOS10: Θ_from_θ

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
# One folder per run (RYF_control, member01, ...). Closed yearly output files and every
# `checkpoint_archive_stride`-th checkpoint are moved to Mediaflux by analysis/storage_watcher.sh.
const run_name = get(ENV, "RUN_NAME", "RYF_control")
const output_path = joinpath("/data/gpfs/projects/punim2499/taimoor/gadi-ensembles/outputs", run_name) * "/"
mkpath(output_path)
const figdir = expanduser("/home/tsohail/uom/ocean-ensembles/figures/")

const Nx = Integer(360 * 6)
const Ny = Integer(180 * 6)
const Nz = Integer(75)
const depth = -5500.0
output_depths = [0, -30, -100, -200, -500, -1000]   # discussion #103 item 1

checkpoint_interval = TimeInterval(5days)
checkpoint_archive_stride = 36            # every 36th checkpoint (180 model days) is kept in Mediaflux
output_interval = AveragedTimeInterval(5days)
yearly_file_splitting() = TimeInterval(365days)   # a new output file every model year; one schedule per writer (TimeInterval is stateful, a shared one splits only the first writer to check it)
callback_iteration_interval = 100
# A separate name from the older RYF_sxtdeg_checkpoint_* files in the same folder,
# so a run of this setup never picks up a checkpoint made on a different grid.
default_checkpoint_prefix = "RYF_sxtdeg_omip_checkpoint"

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

const close_antarctic_shallow_cells = true
const antarctic_closure_latitude = -60.0   # degrees north
const antarctic_closure_depth = 50.0       # m; wet cells shallower than this become land
const global_minimum_depth = 20.0          # m; applied everywhere (latitude < 91°) as a closure

"""
    close_shallow_antarctic_cells!(bottom_height, underlying_grid;
                                   latitude=antarctic_closure_latitude, depth=antarctic_closure_depth)

Turn wet columns shallower than `depth` south of `latitude` into land, then drop any wet
pockets this disconnects from the main ocean.
"""
function close_shallow_antarctic_cells!(bottom_height, underlying_grid;
                                        latitude=antarctic_closure_latitude,
                                        depth=antarctic_closure_depth)
    b = Array(interior(bottom_height, :, :, 1))
    φ = Array(φnodes(underlying_grid, Center(), Center()))
    φ = ndims(φ) == 1 ? repeat(reshape(φ, 1, :), size(b, 1), 1) : φ[1:size(b, 1), 1:size(b, 2)]
    closed = (b .< 0) .& (b .> -depth) .& (φ .< latitude)
    b[closed] .= 0
    set!(bottom_height, reshape(b, size(b)..., 1))
    NumericalEarth.Bathymetry.remove_minor_basins!(bottom_height, 1)
    remaining = Array(interior(bottom_height, :, :, 1))
    @info "Closed $(count(closed)) wet columns shallower than $(depth) m south of $(latitude)°" *
          " ($(count((b .< 0) .& (remaining .>= 0))) more cut off as isolated pockets)"
    return bottom_height
end

# 4-connected labels of `wet` (periodic in i, bounded in j); returns labels and component sizes.
function label_wet_components(wet)
    Nxw, Nyw = size(wet)
    labels = zeros(Int, Nxw, Nyw)
    sizes = Int[]
    for I in CartesianIndices(wet)
        (wet[I] && labels[I] == 0) || continue
        push!(sizes, 0)
        n = length(sizes)
        labels[I] = n
        stack = [I]
        while !isempty(stack)
            J = pop!(stack)
            sizes[n] += 1
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                j = J[2] + dj
                1 <= j <= Nyw || continue
                K = CartesianIndex(mod1(J[1] + di, Nxw), j)
                if wet[K] && labels[K] == 0
                    labels[K] = n
                    push!(stack, K)
                end
            end
        end
    end
    return labels, sizes
end

"""
    apply_minimum_depth_keeping_connections!(bottom_height; depth=global_minimum_depth)

Turn wet columns shallower than `depth` into land, except along the shortest chain of those
columns that reconnects each sea the closure would cut off from the main ocean: that chain
is deepened to `depth` instead. Anything still disconnected is removed afterwards.
"""
function apply_minimum_depth_keeping_connections!(bottom_height; depth=global_minimum_depth)
    b = Array(interior(bottom_height, :, :, 1))
    Nxb, Nyb = size(b)
    shallow = (b .< 0) .& (b .> -depth)
    closed = copy(b)
    closed[shallow] .= 0

    labels, sizes = label_wet_components(closed .< 0)
    main = argmax(sizes)
    connected = labels .== main
    reopened = falses(Nxb, Nyb)
    number_reconnected = 0
    reconnected_cells = 0

    for c in sortperm(sizes, rev=true)
        c == main && continue
        connected[findfirst(==(c), labels)] && continue  # already joined through an earlier sill
        # Breadth-first search from this sea through the closed shallow cells (and other
        # disconnected seas) until it reaches the connected ocean.
        parent = Dict{CartesianIndex{2}, CartesianIndex{2}}()
        queue = findall(==(c), labels)
        seen = falses(Nxb, Nyb)
        seen[queue] .= true
        head = 1
        target = nothing
        while head <= length(queue) && isnothing(target)
            J = queue[head]; head += 1
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                j = J[2] + dj
                1 <= j <= Nyb || continue
                K = CartesianIndex(mod1(J[1] + di, Nxb), j)
                seen[K] && continue
                if connected[K]
                    target = J
                    break
                elseif shallow[K] || labels[K] > 0
                    seen[K] = true
                    parent[K] = J
                    push!(queue, K)
                end
            end
        end
        isnothing(target) && continue

        # Deepen the shallow cells on the path and join everything it passed through.
        J = target
        while true
            if shallow[J] && !reopened[J]
                closed[J] = -depth
                reopened[J] = true
                reconnected_cells += 1
            end
            labels[J] > 0 && (connected[labels .== labels[J]] .= true)
            connected[J] = true
            haskey(parent, J) || break
            J = parent[J]
        end
        connected[labels .== c] .= true
        number_reconnected += 1
    end

    set!(bottom_height, reshape(closed, Nxb, Nyb, 1))
    NumericalEarth.Bathymetry.remove_minor_basins!(bottom_height, 1)
    remaining = Array(interior(bottom_height, :, :, 1))
    @info "Minimum depth $(depth) m: closed $(count(shallow) - reconnected_cells) shallow columns; " *
          "reconnected $(number_reconnected) seas by deepening $(reconnected_cells) sill columns to $(depth) m; " *
          "$(count((closed .< 0) .& (remaining .>= 0))) columns still cut off and removed"
    return bottom_height
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
                                   # Oceananigans 0.117+: name only the north side. The grid adds the
                                   # closed south side itself, and the pivot defaults to FPivot.
                                   fold_topology=FaceFolded)

    @info "Defining bottom bathymetry"
    @time bottom_height = regrid_bathymetry(underlying_grid, bathymetry_metadata;
                                            # Kept at 15 so the original bathymetry cache is reused: regridding at 20
                                            # gave different bottoms (13k new ocean cells) and the pickup blew up.
                                            # The 20 m minimum is applied below as a closure instead (2026-11-11).
                                            minimum_depth=15,
                                            interpolation_passes=25,
                                            major_basins=1)

    # Close the rim of artificially shallow cells along the Antarctic ice-shelf fronts (2026-11-11).
    # ETOPO + minimum_depth leaves 15–60 m cells right against the shelves, where sea ice piled up to
    # 10–30 m and brine-driven coastal jets (3–4.5 m/s) blew the run up at Δt = 13, 20 and 30 min.
    close_antarctic_shallow_cells && close_shallow_antarctic_cells!(bottom_height, underlying_grid)
    # Global 20 m minimum depth (as Simone). Shallow cells become land, except the sills that seas
    # such as the Baltic and the Salish Sea need to stay connected, which are deepened to 20 m.
    apply_minimum_depth_keeping_connections!(bottom_height; depth=global_minimum_depth)

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

# Horizontal-only advective inverse timescale: with AVID the vertical part no longer limits Δt.
@inline horizontal_inverse_timescaleᶜᶜᶜ(i, j, k, grid, u, v) =
    abs(u[i, j, k]) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid) + abs(v[i, j, k]) * Δy⁻¹ᶜᶠᶜ(i, j, k, grid)

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


const spike_log_dir = "/data/gpfs/projects/punim2499/taimoor/claude-notes"

"""
    add_spike_logger!(simulation; interval=10, threshold=0.6, max_cells=20)

Every `interval` iterations, compute the column maximum of the horizontal advective CFL
(|u|/Δx + |v|/Δy times Δt) and append to two CSV files in `spike_log_dir`:
`cfl_max.csv` gets the global maximum, `cfl_spikes.csv` every column above `threshold`
(at most `max_cells`, largest first). Each line gives the cell, the level where the
column peaks, position, bottom depth, CFL and u, v there, to target unphysical regions.
"""
function add_spike_logger!(simulation; interval=10, threshold=0.6, max_cells=20)
    ocean_model = simulation.model.ocean.model
    grid = ocean_model.grid
    u, v, _ = ocean_model.velocities
    Nxg, Nyg, Nzg = size(grid)

    operation = KernelFunctionOperation{Center, Center, Center}(horizontal_inverse_timescaleᶜᶜᶜ, grid, u, v)
    column_maximum = Field(Reduction(maximum!, operation, dims=3))

    cpu_grid = on_architecture(CPU(), grid.underlying_grid)
    λ = Array(λnodes(cpu_grid, Center(), Center()))
    φ = Array(φnodes(cpu_grid, Center(), Center()))
    zc = Array(znodes(cpu_grid, Center()))
    bottom = on_architecture(CPU(), grid.immersed_boundary.bottom_height)
    position(A, i, j) = ndims(A) == 1 ? A[i] : A[i, j]

    header = "day,iteration,dt_min,i,j,k,lon,lat,z,bottom_depth,cfl,u,v\n"
    max_file = joinpath(spike_log_dir, "cfl_max.csv")
    spike_file = joinpath(spike_log_dir, "cfl_spikes.csv")
    isfile(max_file) || write(max_file, header)
    isfile(spike_file) || write(spike_file, header)

    function cell_line(sim, i, j, cfl)
        uc = maybe_allow_scalar(() -> [u[i, j, k] for k in 1:Nzg])
        vc = maybe_allow_scalar(() -> [v[i, j, k] for k in 1:Nzg])
        rate = [abs(uc[k]) * Δx⁻¹ᶠᶜᶜ(i, j, k, cpu_grid) + abs(vc[k]) * Δy⁻¹ᶜᶠᶜ(i, j, k, cpu_grid) for k in 1:Nzg]
        k = argmax(rate)
        return @sprintf("%.4f,%d,%.1f,%d,%d,%d,%.3f,%.3f,%.1f,%.1f,%.3f,%.3f,%.3f\n",
                        sim.model.clock.time / 86400, sim.model.clock.iteration, sim.Δt / 60,
                        i, j, k, position(λ, i, j), position(φ, i, j), zc[k], -bottom[i, j, 1],
                        cfl, uc[k], vc[k])
    end

    function log_spikes(sim)
        try
            compute!(column_maximum)
            cfl = host_interior_copy(column_maximum)[:, :, 1] .* sim.Δt
            cfl[.!isfinite.(cfl)] .= 0
            m, I = findmax(cfl)
            open(io -> write(io, cell_line(sim, I[1], I[2], m)), max_file, "a")
            spikes = findall(cfl .> threshold)
            if !isempty(spikes)
                spikes = spikes[sortperm(cfl[spikes], rev=true)][1:min(end, max_cells)]
                lines = join(cell_line(sim, J[1], J[2], cfl[J]) for J in spikes)
                open(io -> write(io, lines), spike_file, "a")
            end
        catch err
            @warn "CFL spike logging failed" err
        end
        return nothing
    end

    add_callback!(simulation, log_spikes, IterationInterval(interval); name=:cfl_spike_logger)
    return nothing
end

"""
    add_step_diagnostics!(simulation; file=joinpath(spike_log_dir, "step_diagnostics.log"))

Debugging aid (not added by default): after every iteration, log the extremes of S, T, |u|, |v|,
|w|, η, ice thickness and ice speed with their cell indices, to find what blows up first.
"""
function add_step_diagnostics!(simulation; file=joinpath(spike_log_dir, "step_diagnostics.log"),
                               every_step_for=50, then_every=10)
    first_iteration = Ref{Int}(-1)
    function diagnose(sim)
        n = sim.model.clock.iteration
        first_iteration[] < 0 && (first_iteration[] = n)
        (n - first_iteration[] < every_step_for || n % then_every == 0) || return nothing
        try
            ocean_model = sim.model.ocean.model
            ice = sim.model.sea_ice.model
            u, v, w = ocean_model.velocities
            T, S = ocean_model.tracers.T, ocean_model.tracers.S
            ocean = .!iszero.(host_interior_copy(S))
            line = @sprintf("it %d day %.4f Δt %.1f min", sim.model.clock.iteration, sim.model.clock.time / 86400, sim.Δt / 60)
            for (name, f) in (("S", S), ("T", T))
                a = host_interior_copy(f); a[.!ocean] .= NaN
                mn, imn = findmin(x -> isnan(x) ? Inf : x, a); mx, imx = findmax(x -> isnan(x) ? -Inf : x, a)
                line *= @sprintf(" | %s min %.3f %s max %.3f %s", name, mn, Tuple(imn), mx, Tuple(imx))
            end
            for (name, f) in (("u", u), ("v", v), ("w", w), ("η", ocean_model.free_surface.displacement),
                              ("h_ice", ice.ice_thickness), ("u_ice", ice.velocities.u), ("v_ice", ice.velocities.v))
                a = host_interior_copy(f)
                m, I = findmax(x -> isfinite(x) ? abs(x) : Inf, a)
                line *= @sprintf(" | %s %.3e %s", name, m, Tuple(I))
            end
            open(io -> write(io, line * "\n"), file, "a")
        catch err
            @warn "step diagnostics failed" err
        end
        return nothing
    end
    add_callback!(simulation, diagnose, IterationInterval(1); name=:step_diagnostics)
    return nothing
end

"""
    add_column_diagnostics!(simulation, columns; file=joinpath(spike_log_dir, "column_diagnostics.log"))

Debugging aid: after every iteration, write full profiles (T, S, u, v, w, CATKE κc and e) for each
(i, j) in `columns`, plus the surface T and S fluxes and the sea-ice state there.
"""
function add_column_diagnostics!(simulation, columns; file=joinpath(spike_log_dir, "column_diagnostics.log"))
    ocean_model = simulation.model.ocean.model
    grid = ocean_model.grid
    Nzg = size(grid, 3)
    function diagnose(sim)
        try
            u, v, w = ocean_model.velocities
            T, S, e = ocean_model.tracers.T, ocean_model.tracers.S, ocean_model.tracers.e
            κc = ocean_model.closure_fields[1].κc
            JT = T.boundary_conditions.top.condition
            JS = S.boundary_conditions.top.condition
            ice = sim.model.sea_ice.model
            io = IOBuffer()
            for (i, j) in columns
                col(f) = maybe_allow_scalar(() -> [f[i, j, k] for k in 1:Nzg])
                Tc, Sc, uc, vc, wc, κcol, ec = col(T), col(S), col(u), col(v), col(w), col(κc), col(e)
                fluxes = maybe_allow_scalar(() -> (JT isa AbstractField ? JT[i, j, 1] : NaN, JS isa AbstractField ? JS[i, j, 1] : NaN,
                                                    ice.ice_thickness[i, j, 1], ice.ice_concentration[i, j, 1]))
                @printf(io, "it %d day %.4f col (%d,%d) JT %.3e JS %.3e h_ice %.2f a_ice %.2f\n",
                        sim.model.clock.iteration, sim.model.clock.time / 86400, i, j, fluxes...)
                for k in Nzg:-1:1
                    iszero(Sc[k]) && continue
                    @printf(io, "  k %2d T %7.3f S %8.3f u %7.3f v %7.3f w %10.3e κc %9.3e e %9.3e\n",
                            k, Tc[k], Sc[k], uc[k], vc[k], wc[k], κcol[k], ec[k])
                end
            end
            open(f -> write(f, take!(io)), file, "a")
        catch err
            @warn "column diagnostics failed" err
        end
        return nothing
    end
    add_callback!(simulation, diagnose, IterationInterval(1); name=:column_diagnostics)
    return nothing
end

# Verbatim copy of Simone's TotalWaterNormalization (ClimaOcean ss/omip-simulations src/total_water.jl;
# NumericalEarth PR #921). Replace with `NumericalEarth.TotalWaterNormalization` once #921 merges.
#####
##### Holding the water of the ocean, sea ice and snow at its initial value
#####

"""
    TotalWaterNormalization(coupled_model)

A callback that holds the total water of the ocean, sea ice and snow at its initial value. A forced configuration
cannot close its surface water budget, because precipitation and runoff are prescribed while evaporation follows
the model state, so most OMIP-2 models remove the net imbalance at every step [Tsujino et al. (2020)](@cite tsujino2020evaluation).
The free surface absorbs the excess, spread uniformly over the wet area, and the temperature and salinity of each
column are rescaled by the same factor so that their content is unchanged.
"""
struct TotalWaterNormalization{H, O, I, N, A, FT}
    surface_height :: H
    ocean_water :: O
    sea_ice_water :: I
    snow_water :: N
    reference_water :: A
    total_water :: A
    correction :: A
    sea_ice_density_ratio :: FT
    snow_density_ratio :: FT
    wet_area :: FT
end

@inline constant_value(ρ) = ρ
@inline constant_value(ρ::ConstantField) = ρ.constant

function TotalWaterNormalization(coupled_model)
    grid = coupled_model.ocean.model.grid
    FT = eltype(grid)
    sea_ice = coupled_model.sea_ice.model
    ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density

    unit_area = Field{Center, Center, Nothing}(grid)
    set!(unit_area, 1)
    wet_area = only(Array(interior(compute!(Field(Integral(unit_area))))))

    surface_height = Field{Center, Center, Nothing}(grid)
    ocean_water = Field(Integral(surface_height))
    sea_ice_water = Field(Integral(sea_ice.ice_thickness * sea_ice.ice_concentration))
    snow_water = Field(Integral(sea_ice.snow_thickness * sea_ice.ice_concentration))

    reference_water, total_water, correction = (similar(interior(ocean_water)) for _ in 1:3)

    normalization = TotalWaterNormalization(surface_height, ocean_water, sea_ice_water, snow_water,
                                            reference_water, total_water, correction,
                                            convert(FT, constant_value(sea_ice.sea_ice_density) / ρᵒᶜ),
                                            convert(FT, constant_value(sea_ice.snow_density) / ρᵒᶜ),
                                            convert(FT, wet_area))

    compute_total_water!(normalization, coupled_model)
    reference_water .= total_water

    return normalization
end

# Freshwater-equivalent volume of the ocean, sea ice and snow, up to the constant volume below the free surface
function compute_total_water!(normalization, coupled_model)
    # The free surface displacement lives on a grid with extended halos, whose reductions count them as area
    interior(normalization.surface_height) .= interior(coupled_model.ocean.model.free_surface.displacement)

    normalization.total_water .= interior(compute!(normalization.ocean_water)) .+
        normalization.sea_ice_density_ratio .* interior(compute!(normalization.sea_ice_water)) .+
        normalization.snow_density_ratio .* interior(compute!(normalization.snow_water))

    return normalization.total_water
end

@kernel function _correct_sea_level!(η, T, S, grid, δ, kᴺ)
    i, j = @index(Global, NTuple)
    h = static_column_depthᶜᶜᵃ(i, j, grid)
    @inbounds ηⁿ = η[i, j, kᴺ]
    @inbounds δᵢ = δ[1, 1, 1]
    H = h + ηⁿ
    correctable = (h > 0) & (H - δᵢ > 0)
    f = ifelse(correctable, H / (H - δᵢ), one(grid))

    for k in 1:size(grid, 3)
        @inbounds T[i, j, k] *= f
        @inbounds S[i, j, k] *= f
    end

    @inbounds η[i, j, kᴺ] = ηⁿ - ifelse(correctable, δᵢ, zero(grid))
end

function (normalization::TotalWaterNormalization)(simulation)
    coupled_model = simulation.model
    ocean = coupled_model.ocean.model
    grid = ocean.grid
    Nx, Ny, Nz = size(grid)

    compute_total_water!(normalization, coupled_model)
    normalization.correction .= (normalization.total_water .- normalization.reference_water) ./ normalization.wet_area

    launch!(architecture(grid), grid, KernelParameters(1:Nx, 1:Ny), _correct_sea_level!,
            ocean.free_surface.displacement, ocean.tracers.T, ocean.tracers.S, grid, normalization.correction, Nz + 1)

    update_zstar_scaling!(grid, ocean.free_surface.displacement)

    return nothing
end

Oceananigans.prognostic_state(normalization::TotalWaterNormalization) =
    (; reference_water = only(Array(normalization.reference_water)))

function Oceananigans.restore_prognostic_state!(normalization::TotalWaterNormalization, state)
    normalization.reference_water .= state.reference_water
    return normalization
end

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
        ocean_grid = sim.model.ocean.model.grid
        horizontal_cfl = sim.Δt * maximum(KernelFunctionOperation{Center, Center, Center}(horizontal_inverse_timescaleᶜᶜᶜ, ocean_grid, u, v))
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
        msg9 = @sprintf("advective_cfl: %.2f, horizontal_cfl: %.2f\n", advective_cfl, horizontal_cfl)
        @info msg1 * msg2 * msg3 * msg4 * msg5 * msg6 * msg7 * msg8 * msg9
        # Locate horizontal-CFL hotspots (rare, so the extra field is only built when needed), to decide
        # where a targeted viscosity (cf. Simone's narrow-strait ν = Δ²/τ) would be required for longer Δt.
        if horizontal_cfl > 0.45
            try
                hcfl = Field(KernelFunctionOperation{Center, Center, Center}(horizontal_inverse_timescaleᶜᶜᶜ, ocean_grid, u, v))
                compute!(hcfl)
                val, I = findmax_interior_field(hcfl)
                i, j, k = Tuple(I)
                λ, φ, z = maybe_allow_scalar(() -> (λnode(i, j, k, ocean_grid, Center(), Center(), Center()),
                                                     φnode(i, j, k, ocean_grid, Center(), Center(), Center()),
                                                     znode(i, j, k, ocean_grid, Center(), Center(), Center())))
                uu, vv = maybe_allow_scalar(() -> (u[i, j, k], v[i, j, k]))
                line = @sprintf("day %.3f it %d Δt %s hcfl %.2f at (i,j,k)=(%d,%d,%d) lon %.2f lat %.2f z %.1f m u %.2f v %.2f\n",
                                sim.model.clock.time / 86400, iteration, prettytime(sim.Δt), val * sim.Δt, i, j, k, λ, φ, z, uu, vv)
                @info "CFL hotspot: " * line
                open(io -> write(io, line), "/data/gpfs/projects/punim2499/taimoor/claude-notes/cfl_hotspots.log", "a")
            catch err
                @warn "CFL hotspot location failed" err
            end
        end
        wall_time[] = current_wall_time
        return nothing
    end

    add_callback!(simulation, progress, callback_interval)
    return nothing
end
#####
##### Output suite: items 1–6 of github.com/taimoorsohail/ocean-ensembles/discussions/103
#####
##### 1. T, S, u, v, w slices at 0, 30, 100, 200, 500, 1000 m          5-day averages
##### 2. SSH, net heat flux and net freshwater flux into the ocean      5-day averages
##### 3. Mixed layer depth (de Boyer Montégut 2004, Δσ = 0.03 kg m⁻³)    5-day averages
##### 4. Sea-ice concentration and extent (ℵ > 0.15, per hemisphere)     (365/12)-day averages
##### 5. Global ∫Θ dV, ∫S dV and volume                                 5-day averages
##### 6. ∫Θ dz and ∫S dz from the surface to 10, 30, 100, 200, 500, 1000 m, and 1000 m to the bottom   5-day averages
#####
##### Heat and salt contents are stored as raw integrals, as in Simone's OMIP diagnostics: multiply by
##### ρ₀ cₚ (heat, J) or ρ₀ / 1000 (salt, kg) offline. Global integrals use `Integral`, not `Average`,
##### because `Average` fixes its volume at construction, which is wrong on a z-star grid.

const sea_ice_output_interval = AveragedTimeInterval(365days / 12)
const content_depths = (10, 30, 100, 200, 500, 1000)
const sea_ice_extent_threshold = 0.15

# ∫ c dz from the free surface down to depth D (m), counting partial cells and stopping at the sea floor.
@inline function surface_to_depth_integralᶜᶜᵃ(i, j, k, grid, c, D)
    Nzg = size(grid, 3)
    ∫c = zero(grid)
    z = zero(grid)
    for k′ = Nzg:-1:1
        Δzₖ = Oceananigans.Operators.Δzᶜᶜᶜ(i, j, k′, grid) * !immersed_cell(i, j, k′, grid)
        δ = min(Δzₖ, max(zero(grid), D - z))
        ∫c += δ * @inbounds c[i, j, k′]
        z += Δzₖ
    end
    return ∫c
end

# 1 where the ice concentration exceeds the extent threshold in the chosen hemisphere (+1 north, -1 south)
@inline function sea_ice_extent_indicatorᶜᶜᵃ(i, j, k, grid, ℵ, threshold, hemisphere)
    φ = φnode(i, j, 1, grid, Center(), Center(), Center())
    in_extent = (@inbounds ℵ[i, j, 1] > threshold) & (sign(φ) == hemisphere)
    return ifelse(in_extent, one(grid), zero(grid))
end

@inline unit_valueᶜᶜᶜ(i, j, k, grid) = one(grid)

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
    @info "Defining run-dependent output writers for run $run_id_leading (discussion #103 items 1–6)"
    sea_ice_model = simulation.model.sea_ice.model
    T, S = ocean.model.tracers.T, ocean.model.tracers.S
    suffix = "_sxtdeg_RYF_run" * run_id_leading

    # Items 1–3 and 6 share one 5-day writer, so each file stores the (1.5 GB) grid once.
    # 1. Slices of T, S, u, v, w, named e.g. T_0030m
    slice_names = Symbol[]
    slice_fields = []
    for (spec, output_depth) in zip(slice_output_specs(grid), output_depths)
        for (name, c) in pairs(merge((; T, S), ocean.model.velocities))
            push!(slice_names, Symbol(name, "_", lpad(string(abs(output_depth)), 4, '0'), "m"))
            push!(slice_fields, view(c, :, :, spec.ind_pln))
        end
    end
    slice_outputs = NamedTuple{Tuple(slice_names)}(Tuple(slice_fields))

    # 2–3. SSH, net heat and freshwater fluxes into the ocean, mixed layer depth.
    # net_ocean_heat_flux leaves out the shortwave that TwoColorRadiation absorbs below the surface,
    # which it holds as the downward temperature flux entering the water, so it is subtracted here (positive up).
    ρᵒᶜcᵒᶜ = simulation.model.interfaces.ocean_properties.reference_density * simulation.model.interfaces.ocean_properties.heat_capacity
    penetrating_shortwave = NumericalEarth.Oceans.get_radiative_forcing(ocean).surface_flux
    surface_outputs = (; ssh=ocean.model.free_surface.displacement,
                       heat_flux=Field(net_ocean_heat_flux(simulation.model) - ρᵒᶜcᵒᶜ * penetrating_shortwave),
                       fw_flux=Field(net_ocean_freshwater_flux(simulation.model)),
                       mld=MixedLayerDepthField(ocean.model.buoyancy, grid, ocean.model.tracers))

    # Heat budget terms, saved separately (heat_flux = surface_heat_flux + frazil_heat_flux - penetrating_shortwave_flux,
    # and leaves out freshwater_heat_flux). freshwater_mass_flux is the full P - E + R + ice flux; fw_flux holds only the sea-ice part.
    T_top = T.boundary_conditions.top.condition
    freshwater = NumericalEarth.Oceans.freshwater_exchange(T_top)
    ρᵒᶜ = simulation.model.interfaces.ocean_properties.reference_density
    budget_outputs = (; surface_heat_flux=Field(ρᵒᶜcᵒᶜ * NumericalEarth.Diagnostics.flux_field(T_top)), # W m⁻², positive up
                      penetrating_shortwave_flux=Field(ρᵒᶜcᵒᶜ * penetrating_shortwave),               # W m⁻², positive down
                      frazil_heat_flux=NumericalEarth.Diagnostics.frazil_heat_flux(simulation.model),  # W m⁻², as in heat_flux
                      freshwater_heat_flux=Field(ρᵒᶜcᵒᶜ * freshwater.content_flux),                   # W m⁻², positive into the ocean
                      freshwater_mass_flux=Field(ρᵒᶜ * freshwater.carrying_flux))                     # kg m⁻² s⁻¹, positive into the ocean
    surface_outputs = merge(surface_outputs, budget_outputs)

    # Air-sea heat flux terms over open water, in W m⁻², positive UP (out of the ocean), each already multiplied
    # by the open-water fraction (1 - ice concentration) so they are averages over the whole grid cell, like OM2's.
    # NumericalEarth stores the downward radiation already multiplied by (1 - ice concentration), but the emitted
    # longwave and the turbulent fluxes are stored per unit area of open water, so those are multiplied here.
    #   net shortwave absorbed     = shortwave_down_absorbed   (negative: heat into the ocean)
    #   net longwave               = longwave_up + longwave_down_absorbed
    #   sensible + latent          = sensible_heat_flux + latent_heat_flux
    # Check: the five terms add up to about surface_heat_flux - penetrating_shortwave_flux. The two differ by the
    # ice-ocean heat flux, which surface_heat_flux also contains.
    radiative = simulation.model.radiation.interface_fluxes.ocean
    turbulent = simulation.model.interfaces.atmosphere_ocean_interface.fluxes
    open_water = 1 - sea_ice_model.ice_concentration
    flux_term_outputs = (; shortwave_down_absorbed=Field(1 * radiative.downwelling_shortwave),
                         longwave_down_absorbed=Field(1 * radiative.downwelling_longwave),
                         longwave_up=Field(open_water * radiative.upwelling_longwave),
                         sensible_heat_flux=Field(open_water * turbulent.sensible_heat),
                         latent_heat_flux=Field(open_water * turbulent.latent_heat))
    surface_outputs = merge(surface_outputs, flux_term_outputs)

    # 4. Sea-ice concentration and extent
    ℵ = sea_ice_model.ice_concentration
    extent(hemisphere) = Field(Integral(KernelFunctionOperation{Center, Center, Nothing}(sea_ice_extent_indicatorᶜᶜᵃ, grid, ℵ,
                                                                                         sea_ice_extent_threshold, hemisphere),
                                        dims=(1, 2)))
    sea_ice_outputs = (; ice_concentration=ℵ, extent_north=extent(1), extent_south=extent(-1))

    @time simulation.output_writers[:sea_ice] = JLD2Writer(simulation.model, sea_ice_outputs;
                                                           dir=output_path,
                                                           schedule=sea_ice_output_interval,
                                                           filename="sea_ice_monthly" * suffix,
                                                           file_splitting=yearly_file_splitting(),
                                                           with_halos=false,
                                                           overwrite_files=true,
                                                           array_type=Array{Float32})

    # 5. Global heat and salt contents (raw) and volume
    unit = KernelFunctionOperation{Center, Center, Center}(unit_valueᶜᶜᶜ, grid)
    global_outputs = (; T_integral=Field(Integral(T)), S_integral=Field(Integral(S)), volume=Field(Integral(unit)))

    @time ocean.output_writers[:global_integrals] = JLD2Writer(ocean.model, global_outputs;
                                                               dir=output_path,
                                                               schedule=output_interval,
                                                               filename="integrals_5day" * suffix,
                                                               file_splitting=yearly_file_splitting(),
                                                               overwrite_files=true,
                                                               array_type=Array{Float64})

    # 6. Heat and salt contents (raw) from the surface to each depth in `content_depths`
    column_integral(c, D) = KernelFunctionOperation{Center, Center, Nothing}(surface_to_depth_integralᶜᶜᵃ, grid, c,
                                                                              convert(eltype(grid), D))
    names = Symbol[]
    fields = Field[]
    for D in content_depths
        tag = lpad(string(D), 4, '0') * "m"
        push!(names, Symbol("T_integral_0_", tag)); push!(fields, Field(column_integral(T, D)))
        push!(names, Symbol("S_integral_0_", tag)); push!(fields, Field(column_integral(S, D)))
    end
    push!(names, :T_integral_1000m_bottom); push!(fields, Field(column_integral(T, Inf) - column_integral(T, 1000)))
    push!(names, :S_integral_1000m_bottom); push!(fields, Field(column_integral(S, Inf) - column_integral(S, 1000)))
    column_outputs = NamedTuple{Tuple(names)}(Tuple(fields))

    @time simulation.output_writers[:fields_5day] = JLD2Writer(simulation.model,
                                                               merge(slice_outputs, surface_outputs, column_outputs);
                                                               dir=output_path,
                                                               schedule=output_interval,
                                                               filename="fields_5day" * suffix,
                                                               file_splitting=yearly_file_splitting(),
                                                               with_halos=false,
                                                               overwrite_files=true,
                                                               array_type=Array{Float32})

    return nothing
end

# ECCO4 gives potential temperature θ and practical salinity Sᴾ, but the model's TEOS-10
# equation of state expects conservative temperature Θ and absolute salinity Sᴬ.
# We use Sᴬ ≈ (35.16504 / 35) Sᴾ, which leaves out small regional differences (up to about 0.03 g/kg).
const absolute_salinity_per_practical_salinity = 35.16504 / 35

function convert_ecco_to_teos10!(T, S)
    interior(S) .*= absolute_salinity_per_practical_salinity
    interior(T) .= Θ_from_θ.(interior(S), interior(T))
    return nothing
end

function build_simulation(arch, run_id;
                          add_outputs=true,
                          Δt=5minutes,
                          avid=true,
                          checkpoint_prefix=default_checkpoint_prefix)
    dates = ecco_dates()
    dataset = ECCO4Monthly()
    time_indices_in_memory = 24
    inputs = download_input_data!(dates, dataset)

    @info "Defining grid"
    grid = build_grid(arch, inputs.ETOPOmetadata; halo=(7,7,7))
    @info "Defining conservative surface salinity restoring"
    restoring_rate = 1 / 30days

    # All twelve months stay in memory, so the TEOS-10 conversion below is not lost on reload.
    salinity_restoring = DatasetRestoring(inputs.salinity, grid;
                                          rate=restoring_rate,
                                          time_indices_in_memory=length(inputs.salinity))
    parent(salinity_restoring.field_time_series.data) .*= absolute_salinity_per_practical_salinity
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
                              spread_radius=1.2,  # briefly 3.0 on 2026-11-11; reverted: runoff was not the cause of the Weddell runaway
                              maximum_spread_cells=nothing)
    @info "Defining closures"
    # NumericalEarth's default CATKE, with the limits used in the OMIP prototype.
    catke_closure = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization();
                                             mixing_length=CATKEMixingLength(Cᵇ=0.01),
                                             turbulent_kinetic_energy_equation=CATKEEquation(Cᵂϵ=1.0),
                                             maximum_viscosity=3,
                                             maximum_tracer_diffusivity=3,
                                             maximum_tke_diffusivity=3,
                                             negative_tke_damping_time_scale=10)
    # Background κ following Henyey et al. (1986), as in Simone's ss/omip-prototype (was a constant 1e-5).
    background_closure = VerticalScalarDiffusivity(κ=henyey_diffusivity, ν=1e-4)

    river_closure = river_mouth_vertical_diffusivity(grid, land.river_routing;
                                                    river_mouth_diffusivity=0.1,
                                                    river_mouth_mixing_depth=10)  # briefly 100 on 2026-11-11 and 30 on 2026-11-13 (run 29: did not stop the Amazon S spikes); reverted
    # Grid-scaled biharmonic viscosity ν = Az²/λ on momentum only, as Simone's OMIP code
    # (`νhb`, used there at 1/2° and ORCA1). Added 2026-11-11 to damp the unphysical shelf and
    # strait jets (Labrador, Bering, Torres, Yangtze) that blew the run up; stability number ≈ 32Δt/λ.
    biharmonic_closure = HorizontalScalarBiharmonicDiffusivity(ν=νhb, discrete_form=true,
                                                               parameters=biharmonic_timescale)
    # Grid-scaled biharmonic diffusivity κ = Az²/λ on T and S only (2026-11-11). Damps one-cell salt
    # anomalies (e-folding ~1 step at 45 min for λ = 10 days) that otherwise run away in near-neutral,
    # strongly downwelling columns on steep slopes (Weddell polynya at the Filchner front, Labrador),
    # while leaving 8Δx and larger scales nearly untouched. Stability ≈ 390Δt/(2.5λ) ≈ 0.5 at 45 min.
    tracer_biharmonic_closure = HorizontalScalarBiharmonicDiffusivity(κ=(T=νhb, S=νhb, e=0), discrete_form=true,
                                                                      parameters=tracer_biharmonic_timescale)
    # Tracer biharmonic removed 2026-11-12 (as Simone's 1/4°): along-level mixing was suspected of the 200–1000 m warming drift.
    closure = (catke_closure, background_closure, river_closure, biharmonic_closure)

    @info "Defining free surface"
    # 400 substeps (was 220, then 300): barotropic Courant ≈0.58 at Δt=45min, ≈0.39 at 30min (Ross Sea worst cell).
    free_surface = SplitExplicitFreeSurface(grid; substeps=400)
    # avid=false uses fully explicit vertical advection
    vertical_discretization = avid ? AdaptiveVerticallyImplicitDiscretization(cfl=1) : ExplicitTimeDiscretization()
    @info "Vertical advection" avid vertical_discretization
    momentum_advection = WENOVectorInvariant(time_discretization = vertical_discretization)
    # minimum_buffer_upwind_order=3 (Simone's default) since 2026-11-12; it was 1 (monotone 1st-order upwind
    # next to land/seabed, against coastal overshoots such as S = 61 at the Amazon mouth) but suspected of the drift.
    # S back to 1 on 2026-11-13 (run 30): with 3, S spiked to 52-62 / -6 in the bottom cell of the Amazon and Congo mouths.
    tracer_advection = (T = WENO(order=7,
                                   minimum_buffer_upwind_order=3,
                                   time_discretization=vertical_discretization),
                        # bounds=(0, 100) removed 2026-11-11: the column log showed S-only vertical
                        # redistribution (fresh top / salty slab, column salt conserved, T unaffected)
                        # in the Weddell runaway; S now uses exactly T's scheme.
                        S = WENO(order=7,
                                 minimum_buffer_upwind_order=1,
                                 time_discretization=vertical_discretization))

    # Disabled 2026-11-10: Oceananigans 0.117 no longer extends the free-surface halos, so this rewrote the
    # latest checkpoint's η/U/V to the old extended size in place and broke the restore.
    # align_checkpoint_fs(grid, arch, inputs, free_surface, checkpoint_prefix)

    @info "Loading monthly SeaWiFS chlorophyll for shortwave absorption"
    chlorophyll_dates = (DateTime(2000, 1, 1), DateTime(2000, 12, 1))
    chlorophyll_metadata = Metadata(:chlorophyll;
                                    dataset=SeaWiFSMonthly(),
                                    dates=chlorophyll_dates,
                                    dir=data_path)
    # Repeat the same twelve monthly fields, as in Simone's OMIP setup.
    # Keep all twelve months in memory. With only two, any time step that crosses the middle of a
    # month fails, because the barotropic substeps then need three months at once.
    chlorophyll = FieldTimeSeries(chlorophyll_metadata, grid; time_indices_in_memory=12)
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
    convert_ecco_to_teos10!(ocean.model.tracers.T, ocean.model.tracers.S)

    @info "Creating sea ice model"
    # Ice-ocean drag referenced to the top-6 m mean ocean velocity, as Simone's OMIP builder
    # (McPhee's Cᵢₒ is defined against the under-ice boundary layer; the 2.6 m top cell is dragged
    # along by the ice and under-brakes the pack). Added 2026-11-11 against Antarctic ice piling.
    sea_ice_dynamics = NumericalEarth.SeaIces.sea_ice_dynamics(grid, ocean; sea_ice_ocean_drag_reference_depth=6)
    # Four sub-grid ice thickness categories for heat conduction through ice and snow, as Simone's OMIP default
    # (ICE_CATEGORIES=4, added 2026-11-26). Ice in a cell is spread evenly between zero and twice its mean
    # thickness; thin ice conducts more heat, so conduction is 1 + 1/3 + 1/5 + 1/7 ≈ 1.68 times the value at
    # the mean thickness. This only rescales the conductivity, so old checkpoints still load.
    # Each of the four OMIP changes of 2026-11-26 can be switched off for testing with an environment
    # variable set to "0": OMIP_ITD, OMIP_AO_FLUX, OMIP_AI_FLUX, OMIP_GILL.
    itd_shape = UniformThicknessDistribution(eltype(grid); categories = get(ENV, "OMIP_ITD", "1") == "1" ? 4 : 1)
    snow_surface_temperature = Field{Center, Center, Nothing}(grid)
    snow_thermodynamics = snow_slab_thermodynamics(grid; conductivity = eltype(grid)(0.31), itd_shape,
                                                   top_heat_boundary_condition = PrescribedTemperature(snow_surface_temperature.data))
    sea_ice = sea_ice_simulation(grid, ocean;
                                 advection = ClimaSeaIce.IncrementalRemapping(),
                                 dynamics = sea_ice_dynamics,
                                 timestepper = :ForwardEuler,
                                 internal_heat_flux = ConductiveFlux(eltype(grid); conductivity = 2, itd_shape),
                                 snow_thermodynamics)

    dataset_sea_ice = ECCO4Monthly()
    set!(sea_ice.model,
         h=Metadatum(:sea_ice_thickness; dataset=dataset_sea_ice, dir=data_path),
         ℵ=Metadatum(:sea_ice_concentration; dataset=dataset_sea_ice, dir=data_path))

    @info "Defining Atmospheric state"
    # Surface radiation settings copied from Simone's OMIP prototype (NumericalEarth ss/omip-prototype,
    # experiments/OMIPSimulations/src/atmosphere.jl). The NumericalEarth defaults are an ocean albedo of
    # 0.05, an ocean emissivity of 0.97 and a fixed sea-ice albedo of 0.7. Here the ocean reflects 6% of
    # the sunlight and is a perfect emitter, and the sea-ice albedo follows the CCSM3 scheme: it depends on
    # ice thickness, snow depth and surface temperature (lower near melting and for thin ice).
    hi = sea_ice.model.ice_thickness
    hs = sea_ice.model.snow_thickness
    snow_thermo = sea_ice.model.snow_thermodynamics
    Ts = isnothing(snow_thermo) ? sea_ice.model.ice_thermodynamics.top_surface_temperature :
                                  snow_thermo.top_surface_temperature
    sea_ice_albedo = SeaIceAlbedo(hi, hs, Ts)
    radiation = JRA55PrescribedRadiation(arch; time_indices_in_memory,
                                         ocean_surface   = NumericalEarth.Radiations.SurfaceRadiationProperties(0.06, 1.0),
                                         sea_ice_surface = NumericalEarth.Radiations.SurfaceRadiationProperties(sea_ice_albedo, 1.0))
    @info "Radiation: ocean albedo 0.06, ocean emissivity 1.0, sea-ice albedo $sea_ice_albedo"
    atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory)

    @info "Defining coupled model"
    humidities = saturation_enhanced_humidities(eltype(grid))
    use(name) = get(ENV, name, "1") == "1"
    omip_interface_kw = (; (use("OMIP_AO_FLUX") ? (; atmosphere_ocean_fluxes = corrected_atmosphere_ocean_fluxes(eltype(grid))) : (;))...,
                           (use("OMIP_AI_FLUX") ? (; atmosphere_sea_ice_fluxes = corrected_atmosphere_sea_ice_fluxes(eltype(grid))) : (;))...,
                           # Gill humidity over the ocean only: this NumericalEarth version has no keyword for the
                           # sea-ice humidity, so the ice keeps the plain formula.
                           (use("OMIP_GILL") ? (; atmosphere_ocean_interface_specific_humidity = humidities.ocean) : (;))...)
    @info "OMIP switches" ITD=use("OMIP_ITD") AO_FLUX=use("OMIP_AO_FLUX") AI_FLUX=use("OMIP_AI_FLUX") GILL=use("OMIP_GILL")
    @time coupled_model = OceanSeaIceModel(ocean, sea_ice; land, atmosphere, radiation,
                                                  sea_ice_ocean_heat_flux = corrected_ice_ocean_heat_flux(),
                                                  ice_meltwater_enthalpy = ZeroHeatContentMeltwater(),
                                                  # Simone's OMIP flux formulas (2026-11-26): see the functions at the top
                                                  omip_interface_kw...)
    @info "Air-sea fluxes: Simone's OMIP formulas (COARE profile, gust ≥ 0.5 m/s, wind-dependent roughness, Gill humidity over the ocean); 4 ice thickness categories"

    simulation = Simulation(coupled_model; Δt)

    restoring_callback =
        ConservativeSurfaceFluxRestoringCallback(conservative_salinity_restoring, ocean.model)
    add_callback!(simulation, restoring_callback, IterationInterval(1);
                  name=:conservative_salinity_restoring)

    # Hold ocean + sea ice + snow water fixed (Simone's ClimaOcean `TotalWaterNormalization`, 2026-11-11).
    # The reference is reset after every checkpoint restore in `run_segment!`.
    add_callback!(simulation, TotalWaterNormalization(coupled_model), IterationInterval(1); name=:total_water)

    add_progress_callback!(simulation)
    add_spike_logger!(simulation)

    if add_outputs
        add_run_output_writers!(simulation, ocean, grid, run_id)
    end

    @time simulation.output_writers[:checkpointer] = Checkpointer(coupled_model,
                                                                  schedule=checkpoint_interval,
                                                                  dir=output_path,
                                                                  prefix=checkpoint_prefix,
                                                                  overwrite_files=true,
                                                                  cleanup=true)

    # Hard-link every `checkpoint_archive_stride`-th checkpoint into checkpoints_to_archive/ before
    # `cleanup` deletes it; storage_watcher.sh moves it to Mediaflux. Callbacks run before output
    # writers, so at each checkpoint time the newest file on disk is the previous checkpoint.
    archive_dir = joinpath(output_path, "checkpoints_to_archive")
    function archive_checkpoint(sim)
        previous = round(Int, sim.model.clock.time / checkpoint_interval.interval) - 1
        candidates = checkpoint_candidates(checkpoint_prefix)
        if previous % checkpoint_archive_stride == 0 && !isempty(candidates)
            mkpath(archive_dir)
            link = joinpath(archive_dir, basename(first(candidates)))
            isfile(link) || hardlink(first(candidates), link)
        end
        return nothing
    end
    add_callback!(simulation, archive_checkpoint, TimeInterval(checkpoint_interval.interval); name=:checkpoint_archive)

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
# Background tracer diffusivity following Henyey et al. (1986): 2e-6 m² s⁻¹ at the equator rising to 1e-5 at the poles.
# Same as NumericalEarth ss/omip-prototype, experiments/OMIPSimulations/src/omip_simulation.jl:1305.
const biharmonic_timescale = 40days
const tracer_biharmonic_timescale = 10days
@inline νhb(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, λ) = Oceananigans.Operators.Az(i, j, k, grid, ℓx, ℓy, ℓz)^2 / λ

@inline henyey_diffusivity(x, y, z, t) = max(2e-6, 1e-5 * abs(sind(y)))

@inline ocean_cell_indicator(i, j, k, grid) = ifelse(immersed_cell(i, j, k, grid), 0.0, 1.0)

# A CPU array of size (Nx, Ny, Nz) that is true in ocean cells and false in land or sea-floor cells.
# The work is done on the GPU, then copied to the CPU.
function ocean_cells(grid)
    ocean_indicator = Field(KernelFunctionOperation{Center, Center, Center}(ocean_cell_indicator, grid))
    Oceananigans.Fields.compute!(ocean_indicator)
    mask = host_interior_copy(ocean_indicator) .== 1
    free_device_memory!(ocean_indicator)
    return mask
end

# Memory-light host copies: copying a whole contiguous parent array needs no temporary GPU
# buffer, whereas Array(interior(field)) of a GPU field first makes a GPU copy of the view.
function host_interior_copy(field)
    data = Array(parent(field))
    H = (size(data) .- size(field)) .÷ 2
    return data[H[1]+1:H[1]+size(field, 1), H[2]+1:H[2]+size(field, 2), H[3]+1:H[3]+size(field, 3)]
end

function set_interior_from_host!(field, values)
    data = Array(parent(field))
    H = (size(data) .- size(field)) .÷ 2
    data[H[1]+1:H[1]+size(field, 1), H[2]+1:H[2]+size(field, 2), H[3]+1:H[3]+size(field, 3)] .= values
    copyto!(parent(field), data)
    return field
end

free_device_memory!(field) = nothing
if isdefined(Main, :CUDA)
    @eval free_device_memory!(field) =
        (parent(field) isa CUDA.CuArray && CUDA.unsafe_free!(parent(field)); nothing)
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

Columns that gained fewer new cells than they had old ones keep their old water, so an
old cell can still be denser than the filled cell under it. Those columns are mixed
until stable with `mix_unstable_new_cell_columns!`.

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

    salinity = host_interior_copy(tracers.S)
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
    values = [name == :S ? salinity : host_interior_copy(tracers[name]) for name in tracer_names]
    fill_from_old_ocean!(values, ocean, refill_cell; maximum_passes)

    @info "Filled new ocean cells after pickup" number_of_new_cells number_of_deepened_columns

    # Old cells that are denser than the filled cell below them would sink violently.
    # Mix those columns until they are stable, keeping all of their heat and salt.
    if :T in tracer_names && :S in tracer_names
        temperature = values[findfirst(==(:T), tracer_names)]
        salinity = values[findfirst(==(:S), tracer_names)]
        equation_of_state = ocean_model.buoyancy.formulation.equation_of_state
        z_faces = Array(Oceananigans.Grids.rnodes(grid.underlying_grid, Face()))
        mix_unstable_new_cell_columns!(temperature, salinity, ocean, refill_cell, z_faces, equation_of_state)
    end

    for (n, name) in enumerate(tracer_names)
        set_interior_from_host!(tracers[name], values[n])
    end

    # Sea-surface height, as a single level of size (Nx, Ny, 1).
    # A column is new if it is ocean now but none of its cells were ocean in the checkpoint.
    ocean_column = any(ocean, dims=3)
    new_column = ocean_column .& .!any(ocean .& .!new_cell, dims=3)
    number_of_new_columns = count(new_column)

    number_of_new_columns == 0 && return nothing

    η = ocean_model.free_surface.displacement
    η_values = [host_interior_copy(η)]
    fill_from_old_ocean!(η_values, ocean_column, new_column; maximum_passes)
    set_interior_from_host!(η, η_values[1])

    @info "Filled sea-surface height in new ocean columns after pickup" number_of_new_columns

    return nothing
end

"""
    mix_unstable_new_cell_columns!(T, S, ocean, new_cell, z_faces, equation_of_state)

Find columns where an old ocean cell sits directly on a `new_cell` and is denser than it,
then mix each of those columns until it is stable (a convective adjustment). Two cells
are compared at the depth of the face between them, so pressure does not count.
Unstable neighbouring layers are merged into one layer with the thickness-weighted mean
temperature and salinity, so the column keeps all of its heat and salt.

At the Ross Ice Shelf front, for example, very salty water at the bottom of a 13-level
shelf column sat on one new cell of lighter filled water, and the run blew up after
about 700 steps.

`T`, `S`, `ocean` and `new_cell` have size (Nx, Ny, Nz) with k = 1 at the bottom.
`z_faces` holds the Nz + 1 reference face heights.
"""
function mix_unstable_new_cell_columns!(T, S, ocean, new_cell, z_faces, equation_of_state)
    Nx_grid, Ny_grid, Nz_grid = size(ocean)
    ρ′ = Oceananigans.BuoyancyFormulations.ρ′
    Δz = [z_faces[k+1] - z_faces[k] for k in 1:Nz_grid]

    number_of_mixed_columns = 0
    number_of_mixed_cells = 0
    largest_inversion = 0.0
    largest_inversion_cell = (0, 0, 0)

    for j in 1:Ny_grid, i in 1:Nx_grid
        # Is there an old cell (k + 1) that is denser than the new cell (k) under it?
        unstable = false
        for k in 1:Nz_grid-1
            (ocean[i, j, k] && new_cell[i, j, k] && ocean[i, j, k+1] && !new_cell[i, j, k+1]) || continue
            Z = z_faces[k+1]
            inversion = ρ′(T[i, j, k+1], S[i, j, k+1], Z, equation_of_state) -
                        ρ′(T[i, j, k],   S[i, j, k],   Z, equation_of_state)
            if inversion > 0
                unstable = true
                if inversion > largest_inversion
                    largest_inversion = inversion
                    largest_inversion_cell = (i, j, k + 1)
                end
            end
        end
        unstable || continue

        # Convective adjustment, working down from the top.
        # Each layer is (T, S, thickness, top level, bottom level).
        layers = Tuple{Float64, Float64, Float64, Int, Int}[]
        for k in Nz_grid:-1:1
            ocean[i, j, k] || continue
            push!(layers, (T[i, j, k], S[i, j, k], Δz[k], k, k))

            # Merge the newest layer into the one above while the one above is denser.
            while length(layers) >= 2
                Tu, Su, hu, ktop, _ = layers[end-1]
                Tl, Sl, hl, kupper, kbottom = layers[end]
                Z = z_faces[kupper+1]
                ρ′(Tu, Su, Z, equation_of_state) > ρ′(Tl, Sl, Z, equation_of_state) || break
                h = hu + hl
                pop!(layers)
                layers[end] = ((hu * Tu + hl * Tl) / h, (hu * Su + hl * Sl) / h, h, ktop, kbottom)
            end
        end

        for (Tm, Sm, _, ktop, kbottom) in layers
            ktop == kbottom && continue
            for k in kbottom:ktop
                T[i, j, k] = Tm
                S[i, j, k] = Sm
            end
            number_of_mixed_cells += ktop - kbottom + 1
        end
        number_of_mixed_columns += 1
    end

    @info "Mixed columns with dense old water on top of new cells" number_of_mixed_columns number_of_mixed_cells largest_inversion largest_inversion_cell

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
    # Checkpoints from before 2026-11-12 hold a fifth closure (the tracer biharmonic, which has no state)
    number_of_closures = length(simulation.model.ocean.model.closure_fields)
    # (with check_bottom_height, run_segment! has already compared the saved sea floor with the grid)
    if check_bottom_height
        state = without_extra_closure_fields(load_checkpoint_state(filepath), number_of_closures)
        Oceananigans.restore_prognostic_state!(simulation, state)
        return nothing
    end

    @info "Picking up simulation from checkpoint file $(abspath(filepath)) without checking the bottom height"
    state = without_saved_bottom_height(load_checkpoint_state(filepath))
    state = without_extra_closure_fields(state, number_of_closures)
    Oceananigans.restore_prognostic_state!(simulation, state)
    return nothing
end

without_extra_closure_fields(state, n) = state

function without_extra_closure_fields(state::NamedTuple, n)
    values = map(name -> name == :closure_fields && state[name] isa Tuple && length(state[name]) > n ?
                         state[name][1:n] : without_extra_closure_fields(state[name], n), keys(state))
    return NamedTuple{keys(state)}(values)
end

# Oceananigans only checks the bottom height when the checkpoint has one saved,
# so removing every saved `bottom_height` skips the check.
without_saved_bottom_height(state) = state

function without_saved_bottom_height(state::NamedTuple)
    names = filter(name -> name != :bottom_height, keys(state))
    values = map(name -> without_saved_bottom_height(state[name]), names)
    return NamedTuple{names}(values)
end

function clear_sea_ice_on_land!(sea_ice_model, grid)
    bottom = on_architecture(CPU(), grid.immersed_boundary.bottom_height)
    Nx_grid, Ny_grid, _ = size(grid)
    land = [bottom[i, j, 1] >= 0 for i in 1:Nx_grid, j in 1:Ny_grid]
    for name in (:ice_thickness, :ice_concentration, :snow_thickness)
        field = getproperty(sea_ice_model, name)
        field isa Field || continue
        values = host_interior_copy(field)[:, :, 1]
        cleared = count(land .& (values .!= 0))
        values[land] .= 0
        set_interior_from_host!(field, reshape(values, size(values)..., 1))
        cleared > 0 && @info "Cleared $name in $cleared land columns"
    end
    return nothing
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

    # Columns that are land now (e.g. the closed Antarctic rim) can still hold restored sea ice.
    clear_sea_ice_on_land!(simulation.model.sea_ice.model, ocean_model.grid)

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

        # The total-water reference was taken from the initial state when the model was built;
        # hold the water the run has now instead.
        if haskey(simulation.callbacks, :total_water)
            normalization = simulation.callbacks[:total_water].func
            compute_total_water!(normalization, simulation.model)
            normalization.reference_water .= normalization.total_water
        end

        resolved_pickup = false
    end

    @info "Running simulation" state.run_id pickup=resolved_pickup stop_time=prettytime(simulation.stop_time)
    run!(simulation, pickup=resolved_pickup, checkpoint_at_end=true)

    return nothing
end

# To start a fresh run from rest. As in the OMIP prototype, the first 60 days use a short
# time step while the ocean adjusts, then the time step is raised for the rest of the year.
# state = build_simulation(GPU(), 1; add_outputs=true)
# run_segment!(state; pickup=false, Δt=5minutes, stop_time=60days)
# run_segment!(state; pickup=false, Δt=13minutes)

# To rerun from the latest checkpoint. The pickup is reconciled only if the
# checkpoint was saved on a different sea floor (see run_segment!).
# state = nothing
# reclaim_gpu_memory!()
# state = build_simulation(GPU(), 1; add_outputs=true)
# run_segment!(state; pickup=true, Δt=10minutes, stop_time = 10000days)
