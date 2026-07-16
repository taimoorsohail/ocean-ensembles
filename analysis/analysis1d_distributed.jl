using CairoMakie
using Oceananigans  # From local
using NumericalEarth: EarthSystemModels, Oceans
using Statistics
using JLD2
using Glob
using Logging

const ocean_eos = Oceans.TEOS10EquationOfState()
const ρ₀ = isdefined(Oceananigans, :reference_density) ?
           Oceananigans.reference_density(ocean_eos) :
           EarthSystemModels.reference_density(ocean_eos)
const cₚ = isdefined(Oceananigans, :heat_capacity) ?
           Oceananigans.heat_capacity(ocean_eos) :
           EarthSystemModels.heat_capacity(ocean_eos)

output_path = expanduser("/home/tsohail/uom/ocean-ensembles/outputs/saved/")
figdir = expanduser("/home/tsohail/uom/ocean-ensembles/figures/")

keys_interest = ["T_totintegral",
                "S_totintegral",
                "T_vertintegral",
                "S_vertintegral",
                "total_volume_c",
                "vert_volume_c"]

resolution = "sxtdeg"
nframes = nothing
files_integral = glob("combined_global_*tot*$(resolution)*_RYF_run*.jld2", output_path)
files_surface = glob("combined_global_surface_fluxes_$(resolution)*_RYF_run*.jld2", output_path)
files_integral = filter(f -> !occursin("_rank", basename(f)), files_integral)
files_surface = filter(f -> !occursin("_rank", basename(f)), files_surface)

isempty(files_integral) && error("No combined_* integral files found for resolution=$(resolution) in $(output_path)")

run_number(file) = begin
    m = match(r"_run(\d+)\.jld2$", basename(file))
    isnothing(m) ? typemax(Int) : parse(Int, m.captures[1])
end

sort!(files_integral; by = run_number)
sort!(files_surface; by = run_number)

surface_flux_vars = ("heat_flux", "fw_flux")
const SURFACE_FLUX_GC_INTERVAL = 8

function extract_surface_matrix(raw)
    if ndims(raw) == 2
        return raw
    elseif ndims(raw) == 3
        return view(raw, :, :, 1)
    end
    return nothing
end

function underlying_grid(grid)
    return hasproperty(grid, :underlying_grid) ? getproperty(grid, :underlying_grid) : grid
end

function parent_array(source)
    return hasproperty(source, :parent) ? getproperty(source, :parent) : Array(source)
end

function interior_start(source, dim::Int, fallback_halo::Int, interior_size::Int, stored_size::Int)
    if hasproperty(source, :offsets)
        offsets = getproperty(source, :offsets)
        if dim <= length(offsets)
            start = 1 - offsets[dim]
            1 <= start <= stored_size - interior_size + 1 && return start
        end
    end

    stored_size == interior_size && return 1
    start = fallback_halo + 1
    1 <= start <= stored_size - interior_size + 1 && return start
    error("Could not crop stored dimension " * string(stored_size) * " to interior size " * string(interior_size) * ".")
end

function physical_matrix(source, grid; T = Float64)
    Nx, Ny = getproperty(grid, :Nx), getproperty(grid, :Ny)
    Hx, Hy = getproperty(grid, :Hx), getproperty(grid, :Hy)
    data = parent_array(source)
    i0 = interior_start(source, 1, Hx, Nx, size(data, 1))
    j0 = interior_start(source, 2, Hy, Ny, size(data, 2))

    if ndims(data) == 2
        return T.(view(data, i0:i0+Nx-1, j0:j0+Ny-1))
    elseif ndims(data) == 3
        return T.(view(data, i0:i0+Nx-1, j0:j0+Ny-1, 1))
    end

    error("Expected a 2D or 3D stored grid array, got " * string(ndims(data)) * " dimensions.")
end

function cell_area_matrix(grid)
    g = underlying_grid(grid)
    hasproperty(g, :Azᶜᶜᵃ) || error("Serialized grid does not contain center cell areas.")
    return physical_matrix(getproperty(g, :Azᶜᶜᵃ), g; T = Float64)
end

function bottom_height_matrix(grid)
    hasproperty(grid, :immersed_boundary) || return nothing
    source_grid = underlying_grid(grid)
    immersed_boundary = getproperty(grid, :immersed_boundary)
    hasproperty(immersed_boundary, :bottom_height) || return nothing
    bottom_height_field = getproperty(immersed_boundary, :bottom_height)
    hasproperty(bottom_height_field, :data) || return nothing
    return physical_matrix(getproperty(bottom_height_field, :data), source_grid; T = Float32)
end

surface_ocean_mask(bottom_height::Union{Nothing, AbstractMatrix}) = isnothing(bottom_height) ? nothing : BitMatrix(bottom_height .< 0f0)

function surface_flux_integral(flux::AbstractMatrix, areas::AbstractMatrix, wet_mask::Union{Nothing, BitMatrix})
    size(flux) == size(areas) || error("Area shape mismatch: got " * string(size(areas)) * " expected " * string(size(flux)) * ".")
    !isnothing(wet_mask) && size(wet_mask) != size(flux) && error("Mask shape mismatch: got " * string(size(wet_mask)) * " expected " * string(size(flux)) * ".")

    total = 0.0
    @inbounds for idx in eachindex(flux, areas)
        !isnothing(wet_mask) && !wet_mask[idx] && continue
        value = flux[idx]
        isfinite(value) && (total += Float64(value) * areas[idx])
    end
    return total
end

function linear_interpolate_series(query_times, source_times, source_values)
    Nq = length(query_times)
    Ns = length(source_times)
    Ns >= 2 || error("Need at least 2 source points for interpolation.")

    out = Vector{Float64}(undef, Nq)
    j = 1

    for i in 1:Nq
        t = query_times[i]
        while j < Ns - 1 && source_times[j+1] < t
            j += 1
        end

        t₀, t₁ = source_times[j], source_times[j+1]
        y₀, y₁ = source_values[j], source_values[j+1]

        if t₁ == t₀
            out[i] = y₀
        else
            α = (t - t₀) / (t₁ - t₀)
            out[i] = (1 - α) * y₀ + α * y₁
        end
    end

    return out
end

function read_serialized_grid(file)
    haskey(file, "serialized/grid") && return file["serialized/grid"]
    haskey(file, "serialized") || return nothing

    serialized_keys = sort!(String.(collect(keys(file["serialized"]))))
    for key in serialized_keys
        startswith(key, "grid") || continue
        return file["serialized/$key"]
    end

    return nothing
end

function load_serialized_grid(files::Vector{String}; label::AbstractString)
    for (index, file) in enumerate(files)
        grid = with_logger(NullLogger()) do
            jldopen(file, "r") do data
                read_serialized_grid(data)
            end
        end

        isnothing(grid) && continue

        if index > 1
            @info "Falling back to grid metadata from peer combined file." label source = basename(file)
        end

        return grid
    end

    return nothing
end

function infer_surface_flux_sign(run::Int, flux_times, heat_flux_raw, fw_flux_raw, integral_runs)
    run_data = get(integral_runs, run, nothing)
    isnothing(run_data) && return 1.0

    overlap_start = max(first(run_data.times), first(flux_times))
    overlap_end = min(last(run_data.times), last(flux_times))
    overlap_end > overlap_start || return 1.0

    overlap_idx = findall(t -> overlap_start <= t <= overlap_end, run_data.times)
    length(overlap_idx) >= 2 || return 1.0

    dt_in_seconds = vcat(0.0, diff(flux_times))
    heat_flux_cumsum = cumsum(heat_flux_raw .* dt_in_seconds)
    fw_flux_cumsum = cumsum(fw_flux_raw .* dt_in_seconds)

    compare_times = run_data.times[overlap_idx]
    heat_compare = linear_interpolate_series(compare_times, flux_times, heat_flux_cumsum)
    fw_compare = linear_interpolate_series(compare_times, flux_times, fw_flux_cumsum)

    heat_compare .-= heat_compare[1]
    fw_compare .-= fw_compare[1]

    ohc_anomaly = run_data.ohc[overlap_idx] .- run_data.ohc[overlap_idx[1]]
    fw_anomaly = run_data.fw[overlap_idx] .- run_data.fw[overlap_idx[1]]

    plus_score = sum(abs2, ohc_anomaly .- heat_compare) + sum(abs2, fw_anomaly .- fw_compare)
    minus_score = sum(abs2, ohc_anomaly .+ heat_compare) + sum(abs2, fw_anomaly .+ fw_compare)

    inferred_sign = plus_score <= minus_score ? 1.0 : -1.0
    @info "Inferred surface-flux sign convention" run inferred_sign plus_score minus_score overlap_count = length(overlap_idx)
    return inferred_sign
end

time_total = Float64[]
T_total = Float64[]
S_total = Float64[]
V_total = Float64[]

depth = Float64[]
integral_grid = load_serialized_grid(files_integral; label = "integral diagnostics")
integral_grid === nothing && error("No serialized grid found in combined integral files.")
integral_depth_grid = hasproperty(integral_grid, :underlying_grid) ? integral_grid.underlying_grid : integral_grid
append!(depth, integral_depth_grid.z.cᵃᵃᶜ)

integral_by_time = Dict{Float64, NamedTuple{(:run, :T, :S, :V, :Tz, :Sz, :Vz), Tuple{Int, Float64, Float64, Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
integral_run_by_time = Dict{Int, Dict{Float64, NamedTuple{(:T, :S, :V), Tuple{Float64, Float64, Float64}}}}()
integral_replacements = Ref(0)

for file in files_integral
    @info "Processing combined file $(basename(file))"
    run = run_number(file)

    jldopen(file, "r") do data
        timeiters = sort(parse.(Int, keys(data["timeseries/t/"])))
        for iter in timeiters
            t = Float64(data["timeseries/t/$(iter)"])
            record = (
                run = run,
                T = data["timeseries/T_totintegral/$(iter)"][1, 1, 1],
                S = data["timeseries/S_totintegral/$(iter)"][1, 1, 1],
                V = data["timeseries/total_volume_c/$(iter)"][1, 1, 1],
                Tz = collect(vec(data["timeseries/T_vertintegral/$(iter)"][1, 1, 7:end-7])),
                Sz = collect(vec(data["timeseries/S_vertintegral/$(iter)"][1, 1, 7:end-7])),
                Vz = collect(vec(data["timeseries/vert_volume_c/$(iter)"][1, 1, 7:end-7])))

            existing = get(integral_by_time, t, nothing)
            if isnothing(existing) || run >= existing.run
                integral_replacements[] += (!isnothing(existing) && run > existing.run) ? 1 : 0
                integral_by_time[t] = record
            end

            run_integrals = get!(integral_run_by_time, run, Dict{Float64, NamedTuple{(:T, :S, :V), Tuple{Float64, Float64, Float64}}}())
            run_integrals[t] = (T = record.T, S = record.S, V = record.V)
        end
    end
end

sorted_integral_times = sort(collect(keys(integral_by_time)))
N = length(sorted_integral_times)
N > 0 || error("No new timeseries entries found in combined files.")
Lz = length(integral_by_time[first(sorted_integral_times)].Tz)

if any(t -> begin
        entry = integral_by_time[t]
        length(entry.Tz) != Lz || length(entry.Sz) != Lz || length(entry.Vz) != Lz
    end, sorted_integral_times)
    error("Inconsistent vertical vector lengths across timeseries entries.")
end

if length(depth) != Lz
    @warn "Depth length ($(length(depth))) does not match vertical-integral length ($(Lz)); adjusting depth to match."
    if length(depth) > Lz
        depth = depth[1:Lz]
    elseif !isempty(depth)
        depth = collect(range(first(depth), last(depth), length = Lz))
    else
        depth = collect(range(-Lz, 0, length = Lz))
    end
end

sizehint!(time_total, N)
sizehint!(T_total, N)
sizehint!(S_total, N)
sizehint!(V_total, N)
T_z_matrix = Matrix{Float64}(undef, N, Lz)
S_z_matrix = Matrix{Float64}(undef, N, Lz)
V_z_matrix = Matrix{Float64}(undef, N, Lz)

for (i, t) in enumerate(sorted_integral_times)
    entry = integral_by_time[t]
    push!(time_total, t)
    push!(T_total, entry.T)
    push!(S_total, entry.S)
    push!(V_total, entry.V)
    T_z_matrix[i, :] = entry.Tz
    S_z_matrix[i, :] = entry.Sz
    V_z_matrix[i, :] = entry.Vz
end

@info "Merged integral timeseries with run-priority deduplication" unique_steps = length(time_total) replaced_duplicates = integral_replacements[]

integral_runs = Dict{Int, NamedTuple{(:times, :ohc, :fw), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
for (run, run_integrals) in integral_run_by_time
    run_times = sort(collect(keys(run_integrals)))
    T_run = [run_integrals[t].T for t in run_times]
    S_run = [run_integrals[t].S for t in run_times]
    V_run = [run_integrals[t].V for t in run_times]
    integral_runs[run] = (
        times = run_times,
        ohc = ρ₀ .* cₚ .* T_run,
        fw = ρ₀ .* (V_run .- S_run ./ 35)
    )
end

empty!(integral_by_time)
empty!(integral_run_by_time)
GC.gc()

t_all = time_total
time_in_years = t_all ./ (3600 * 24 * 365)

surface_flux_integrals = Dict("heat_flux" => Float64[], "fw_flux" => Float64[])
surface_flux_times = Float64[]
surface_flux_by_time = Dict{Float64, NamedTuple{(:run, :heat_flux, :fw_flux), Tuple{Int, Float64, Float64}}}()
surface_flux_replacements = Ref(0)
surface_grid = load_serialized_grid(files_surface; label = "surface-flux diagnostics")

if surface_grid === nothing
    @info "Using integral grid metadata for surface-flux diagnostics."
    surface_grid = integral_grid
end

surface_areas = cell_area_matrix(surface_grid)
surface_wet_mask = surface_ocean_mask(bottom_height_matrix(surface_grid))

for file in files_surface
    @info "Streaming surface flux diagnostics from $(basename(file))"
    run = run_number(file)

    jldopen(file, "r") do data
        haskey(data, "timeseries/t") || return
        haskey(data, "timeseries/heat_flux") || return
        haskey(data, "timeseries/fw_flux") || return


        timeiters = sort(parse.(Int, keys(data["timeseries/t/"])))
        file_surface_times = Float64[]
        raw_heat_integrals = Float64[]
        raw_fw_integrals = Float64[]

        for (t_idx, iter) in enumerate(timeiters)
            t = Float64(data["timeseries/t/$(iter)"])
            heat_raw = extract_surface_matrix(data["timeseries/heat_flux/$(iter)"])
            fw_raw = extract_surface_matrix(data["timeseries/fw_flux/$(iter)"])
            (heat_raw === nothing || fw_raw === nothing) && continue

            push!(file_surface_times, t)
            push!(raw_heat_integrals, surface_flux_integral(heat_raw, surface_areas, surface_wet_mask))
            push!(raw_fw_integrals, surface_flux_integral(fw_raw, surface_areas, surface_wet_mask))

            if t_idx % SURFACE_FLUX_GC_INTERVAL == 0
                GC.gc(false)
            end
        end

        isempty(file_surface_times) && return

        flux_sign = infer_surface_flux_sign(run, file_surface_times, raw_heat_integrals, raw_fw_integrals, integral_runs)
        @info "Applying inferred surface-flux sign convention" file = basename(file) run flux_sign

        for i in eachindex(file_surface_times)
            t = file_surface_times[i]
            record = (
                run = run,
                heat_flux = flux_sign * raw_heat_integrals[i],
                fw_flux = flux_sign * raw_fw_integrals[i])

            existing = get(surface_flux_by_time, t, nothing)
            if isnothing(existing) || run >= existing.run
                surface_flux_replacements[] += (!isnothing(existing) && run > existing.run) ? 1 : 0
                surface_flux_by_time[t] = record
            end
        end
    end

    GC.gc(false)
end

for t in sort(collect(keys(surface_flux_by_time)))
    push!(surface_flux_times, t)
    push!(surface_flux_integrals["heat_flux"], surface_flux_by_time[t].heat_flux)
    push!(surface_flux_integrals["fw_flux"], surface_flux_by_time[t].fw_flux)
end

@info "Merged streamed surface-flux collections" runs = length(files_surface) variables = collect(surface_flux_vars)
empty!(surface_flux_by_time)
GC.gc()

isempty(surface_flux_times) && error("No streamed surface-flux timesteps found.")
@info "Merged surface-flux timeseries with run-priority deduplication" unique_steps = length(surface_flux_times) replaced_duplicates = surface_flux_replacements[]

dt_in_seconds = vcat(0.0, diff(surface_flux_times))
surface_flux_cumsum = Dict(
    "heat_flux" => cumsum(surface_flux_integrals["heat_flux"] .* dt_in_seconds),
    "fw_flux" => cumsum(surface_flux_integrals["fw_flux"] .* dt_in_seconds))

overlap_start = max(minimum(t_all), minimum(surface_flux_times))
overlap_end = min(maximum(t_all), maximum(surface_flux_times))
(overlap_end > overlap_start) || error("No overlapping time window between integral diagnostics and surface flux timeseries.")

target_overlap_idx = findall(t -> overlap_start <= t <= overlap_end, t_all)
!isempty(target_overlap_idx) || error("No integral times found in overlapping time window.")

t_compare = t_all[target_overlap_idx]
time_compare_years = t_compare ./ (3600 * 24 * 365)
heat_flux_cumsum_compare = linear_interpolate_series(t_compare, surface_flux_times, surface_flux_cumsum["heat_flux"])
fw_flux_cumsum_compare = linear_interpolate_series(t_compare, surface_flux_times, surface_flux_cumsum["fw_flux"])

integral_dt_days = length(t_all) > 1 ? median(diff(t_all)) / 86400 : NaN
flux_dt_days = length(surface_flux_times) > 1 ? median(diff(surface_flux_times)) / 86400 : NaN
@info "Using overlapping time window for flux comparisons (flux cumsum interpolated to integral timestamps)" overlap_count = length(target_overlap_idx) overlap_start overlap_end integral_count = length(t_all) flux_count = length(surface_flux_times) integral_dt_days flux_dt_days

fig = Figure(size = (900, 900))
ax1 = Axis(fig[1, 1], title = "OHC Integral", xlabel = "Time (years)", ylabel = "Energy (J)")
ax2 = Axis(fig[1, 2], title = "Freshwater Content Integral", xlabel = "Time (years)", ylabel = "Freshwater mass (kg)")
ax3 = Axis(fig[2, 1], title = "Cumulative Heat Flux Integral", xlabel = "Time (years)", ylabel = "∫HF dt")
ax4 = Axis(fig[2, 2], title = "Cumulative Freshwater Flux Integral", xlabel = "Time (years)", ylabel = "∫FW dt")
ax5 = Axis(fig[3, 1], title = "OHC - ∫HF dt", xlabel = "Time (years)", ylabel = "Difference (J)")
ax6 = Axis(fig[3, 2], title = "FW Content - ∫FW dt", xlabel = "Time (years)", ylabel = "Difference (kg)")

ohc = ρ₀ * cₚ * T_total
mean_temperature = T_total ./ V_total
fw_content = ρ₀ .* (V_total .- S_total / 35)
mean_salinity = S_total ./ V_total

ohc_anomaly = ohc .- ohc[1]
fw_content_anomaly = fw_content .- fw_content[1]
ohc_anomaly_compare = ohc[target_overlap_idx] .- ohc[target_overlap_idx[1]]
fw_content_anomaly_compare = fw_content_anomaly[target_overlap_idx] .- fw_content_anomaly[target_overlap_idx[1]]
heat_flux_cumsum_compare .-= heat_flux_cumsum_compare[1]
fw_flux_cumsum_compare .-= fw_flux_cumsum_compare[1]

lines!(ax1, time_compare_years, ohc_anomaly_compare, label = "OHC anomaly")
lines!(ax2, time_compare_years, fw_content_anomaly_compare, label = "FW content anomaly")
lines!(ax3, time_compare_years, heat_flux_cumsum_compare, label = "cumsum(∫heat_flux dA · dt)")
lines!(ax4, time_compare_years, fw_flux_cumsum_compare, label = "cumsum(∫fw_flux dA · dt)")

ohc_difference = ohc_anomaly_compare .- heat_flux_cumsum_compare
fw_difference = fw_content_anomaly_compare .- fw_flux_cumsum_compare
lines!(ax5, time_compare_years, ohc_difference, label = "Difference")
lines!(ax6, time_compare_years, fw_difference, label = "Difference")
hlines!(ax5, [0.0], color = :black, linestyle = :dash)
hlines!(ax6, [0.0], color = :black, linestyle = :dash)


figpath_integrated = joinpath(figdir, "integrated_props_$(resolution).png")
save(figpath_integrated, fig, px_per_unit=3)

fig = Figure(size = (800, 500))
ax1 = Axis(fig[1, 1], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
ax2 = Axis(fig[2, 1], title = "Mean Temperature", xlabel = "Time (years)", ylabel = "Temperature (°C)")
ax3 = Axis(fig[1, 2], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
ax4 = Axis(fig[2, 2], title = "Mean Salinity", xlabel = "Time (years)", ylabel = "Salinity (psu)")

allranks_T_z_int_anomaly = T_z_matrix .- T_z_matrix[1, :]'
allranks_S_z_int_anomaly = S_z_matrix .- S_z_matrix[1, :]'
allranks_V_z_int = V_z_matrix

heatmap!(ax1, time_in_years, depth, ρ₀ * cₚ * allranks_T_z_int_anomaly, label = "OHC", colorrange = (-1e21, 1e21), colormap = :bwr)
heatmap!(ax2, time_in_years, depth, ρ₀ * cₚ * allranks_T_z_int_anomaly ./ allranks_V_z_int, label = "Mean Temperature")
heatmap!(ax3, time_in_years, depth, allranks_S_z_int_anomaly ./ (35 * ρ₀), label = "OSC", colorrange = (-1e10, 1e10), colormap = :bwr)
heatmap!(ax4, time_in_years, depth, allranks_S_z_int_anomaly./allranks_V_z_int, label = "Mean Salinity")

ylims!(ax1, -1000, 0)
ylims!(ax2, -1000, 0)
ylims!(ax3, -1000, 0)
ylims!(ax4, -1000, 0)

figpath_integrated_z = joinpath(figdir, "integrated_props_z_$(resolution).png")
save(figpath_integrated_z, fig, px_per_unit=3)

@info "Created figure files:" figpath_integrated figpath_integrated_z

# fig = Figure(size = (800, 600))
# ax1 = Axis(fig[1, 1], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
# lines!(ax1, depth[1], 1035*1000*allranks_T_z_int_anomaly[16,:])
# xlims!(ax1, 0, -1000)
# ylims!(ax1, -1e20, 1e20)
# save(figdir * "test_temp_profile.png", fig, px_per_unit=3)
