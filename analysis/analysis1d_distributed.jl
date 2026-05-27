using CairoMakie
using Oceananigans  # From local
using NumericalEarth: EarthSystemModels, Oceans
using Statistics
using JLD2
using Glob

const ocean_eos = Oceans.TEOS10EquationOfState()
const ρ₀ = isdefined(Oceananigans, :reference_density) ?
           Oceananigans.reference_density(ocean_eos) :
           EarthSystemModels.reference_density(ocean_eos)
const cₚ = isdefined(Oceananigans, :heat_capacity) ?
           Oceananigans.heat_capacity(ocean_eos) :
           EarthSystemModels.heat_capacity(ocean_eos)

output_path = expanduser("/home/tsohail/uom/ocean-ensembles/outputs/")
figdir = expanduser("/home/tsohail/uom/ocean-ensembles/figures/")

function copy_files_to_tempdir(files::Vector{String}; prefix::String)
    copy_dir = mktempdir(; prefix)
    copied = String[]

    try
        for file in files
            dest = joinpath(copy_dir, basename(file))
            cp(file, dest; force = true)
            push!(copied, dest)
        end
    catch
        rm(copy_dir; recursive = true, force = true)
        rethrow()
    end

    @info "Copied analysis inputs." source_files = length(files) copy_dir
    return copied, copy_dir
end

function cleanup_copied_outputs!(copy_dir::Union{Nothing, String})
    if copy_dir !== nothing && isdir(copy_dir)
        rm(copy_dir; recursive = true, force = true)
        @info "Deleted copied analysis inputs." copy_dir
    end
    return nothing
end

keys_interest = ["T_totintegral",
                "S_totintegral",
                "T_vertintegral",
                "S_vertintegral",
                "total_volume_c",
                "vert_volume_c"]

resolution = "sxtdeg"
nframes = nothing
files_integral = glob("global_*tot*$(resolution)*_RYF_run*.jld2", output_path)
files_surface = glob("global_surface_fluxes_$(resolution)*_RYF_run*.jld2", output_path)
files_integral = filter(f -> !occursin("_rank", basename(f)), files_integral)
files_surface = filter(f -> !occursin("_rank", basename(f)), files_surface)

isempty(files_integral) && error("No combined integral files found for resolution=$(resolution) in $(output_path)")

run_number(file) = begin
    m = match(r"_run(\d+)\.jld2$", basename(file))
    isnothing(m) ? typemax(Int) : parse(Int, m.captures[1])
end

sort!(files_integral; by = run_number)
sort!(files_surface; by = run_number)

copied_output_dir = nothing
copied_inputs = unique(vcat(files_integral, files_surface))
copied_files, copied_output_dir = copy_files_to_tempdir(copied_inputs; prefix = "analysis1d_distributed_")
atexit(() -> cleanup_copied_outputs!(copied_output_dir))

files_integral = filter(f -> occursin("tot", basename(f)), copied_files)
files_surface = filter(f -> occursin("surface_fluxes", basename(f)), copied_files)
sort!(files_integral; by = run_number)
sort!(files_surface; by = run_number)

surface_flux_vars = ("heat_flux", "fw_flux")
surface_flux_timeseries = Dict(var => Any[] for var in surface_flux_vars)

for file in files_surface
    @info "Loading surface flux FieldTimeSeries from $(basename(file))"
    for var in surface_flux_vars
        push!(surface_flux_timeseries[var], FieldTimeSeries(file, var))
    end
end

@info "Loaded surface flux FieldTimeSeries collections" runs = length(files_surface) variables = collect(surface_flux_vars)

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

time_total = Float64[]
T_total = Float64[]
S_total = Float64[]
V_total = Float64[]

T_z_total = Vector{Vector{Float64}}()
S_z_total = Vector{Vector{Float64}}()
V_z_total = Vector{Vector{Float64}}()

depth = Vector{Float64}()
integral_by_time = Dict{Float64, NamedTuple{(:run, :T, :S, :V, :Tz, :Sz, :Vz), Tuple{Int, Float64, Float64, Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
integral_replacements = Ref(0)

for file in files_integral
    @info "Processing combined file $(basename(file))"
    run = run_number(file)

    jldopen(file, "r") do data
        isempty(depth) && append!(depth, data["serialized/grid"].underlying_grid.z.cᵃᵃᶜ)

        timeiters = sort(parse.(Int, keys(data["timeseries/t/"])))
        for iter in timeiters
            t = Float64(data["timeseries/t/$(iter)"])
            record = (
                run = run,
                T = data["timeseries/T_totintegral/$(iter)"][1, 1, 1],
                S = data["timeseries/S_totintegral/$(iter)"][1, 1, 1],
                V = data["timeseries/total_volume_c/$(iter)"][1, 1, 1],
                Tz = vec(data["timeseries/T_vertintegral/$(iter)"][1, 1, 7:end-7]),
                Sz = vec(data["timeseries/S_vertintegral/$(iter)"][1, 1, 7:end-7]),
                Vz = vec(data["timeseries/vert_volume_c/$(iter)"][1, 1, 7:end-7]))

            existing = get(integral_by_time, t, nothing)
            if isnothing(existing) || run >= existing.run
                integral_replacements[] += (!isnothing(existing) && run > existing.run) ? 1 : 0
                integral_by_time[t] = record
            end
        end
    end
end

for t in sort(collect(keys(integral_by_time)))
    entry = integral_by_time[t]
    push!(time_total, t)
    push!(T_total, entry.T)
    push!(S_total, entry.S)
    push!(V_total, entry.V)
    push!(T_z_total, entry.Tz)
    push!(S_z_total, entry.Sz)
    push!(V_z_total, entry.Vz)
end

@info "Merged integral timeseries with run-priority deduplication" unique_steps = length(time_total) replaced_duplicates = integral_replacements[]

N = length(time_total)
N > 0 || error("No new timeseries entries found in combined files.")
Lz = length(T_z_total[1])

if any(length.(T_z_total) .!= Lz) || any(length.(S_z_total) .!= Lz) || any(length.(V_z_total) .!= Lz)
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

t_all = copy(time_total)
T_all = reshape(copy(T_total), N, 1)
S_all = reshape(copy(S_total), N, 1)
V_all = reshape(copy(V_total), N, 1)

T_z_matrix = permutedims(reduce(hcat, T_z_total), (2, 1))
S_z_matrix = permutedims(reduce(hcat, S_z_total), (2, 1))
V_z_matrix = permutedims(reduce(hcat, V_z_total), (2, 1))

T_z_all = reshape(T_z_matrix, N, 1, Lz)
S_z_all = reshape(S_z_matrix, N, 1, Lz)
V_z_all = reshape(V_z_matrix, N, 1, Lz)

perm = sortperm(t_all)
t_all = t_all[perm]
T_all = T_all[perm, :]
S_all = S_all[perm, :]
V_all = V_all[perm, :]
T_z_all = T_z_all[perm, :, :]
S_z_all = S_z_all[perm, :, :]
V_z_all = V_z_all[perm, :, :]

time_in_years = t_all ./ (3600*24*365)

surface_flux_integrals = Dict("heat_flux" => Float64[], "fw_flux" => Float64[])
surface_flux_times = Float64[]
surface_flux_by_time = Dict{Float64, NamedTuple{(:run, :heat_flux, :fw_flux), Tuple{Int, Float64, Float64}}}()
surface_flux_replacements = Ref(0)

for run_idx in eachindex(files_surface)
    run = run_number(files_surface[run_idx])
    heat_fts = surface_flux_timeseries["heat_flux"][run_idx]
    fw_fts = surface_flux_timeseries["fw_flux"][run_idx]
    Nt = min(length(heat_fts), length(fw_fts))

    for t_idx in 1:Nt
        @show t_idx
        @show heat_fts
        t = Float64(heat_fts.times[t_idx])

        # Horizontal-only integral: `dims=(1,2)` integrates with dA, not dz.
        heat_integral = compute!(Field(Integral(heat_fts[t_idx], dims = (1, 2))))
        fw_integral = compute!(Field(Integral(fw_fts[t_idx], dims = (1, 2))))
        record = (
            run = run,
            heat_flux = -interior(heat_integral)[1, 1, 1],
            fw_flux = interior(fw_integral)[1, 1, 1])

        existing = get(surface_flux_by_time, t, nothing)
        if isnothing(existing) || run >= existing.run
            surface_flux_replacements[] += (!isnothing(existing) && run > existing.run) ? 1 : 0
            surface_flux_by_time[t] = record
        end
    end
end

for t in sort(collect(keys(surface_flux_by_time)))
    push!(surface_flux_times, t)
    push!(surface_flux_integrals["heat_flux"], surface_flux_by_time[t].heat_flux)
    push!(surface_flux_integrals["fw_flux"], surface_flux_by_time[t].fw_flux)
end

isempty(surface_flux_times) && error("No surface-flux FieldTimeSeries timesteps found.")
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

fig = Figure(size = (900, 700))
ax1 = Axis(fig[1, 1], title = "OHC vs ∫ Heat Flux dt", xlabel = "Time (years)", ylabel = "Energy (J)")
ax3 = Axis(fig[1, 2], title = "OSC vs ∫ Freshwater Flux dt", xlabel = "Time (years)", ylabel = "Freshwater-equivalent (kg)")
ax5 = Axis(fig[2, 1], title = "OHC Comparison Difference", xlabel = "Time (years)", ylabel = "OHC - ∫HFdt")
ax6 = Axis(fig[2, 2], title = "OSC Comparison Difference", xlabel = "Time (years)", ylabel = "OSC - ∫FWdt")
ax2 = Axis(fig[3, 1], title = "Mean Temperature", xlabel = "Time (years)", ylabel = "Temperature (°C)")
ax4 = Axis(fig[3, 2], title = "Mean Salinity", xlabel = "Time (years)", ylabel = "Salinity (psu)")

ohc = ρ₀ * cₚ * sum(T_all, dims=2)[:, 1]
mean_temperature = sum(T_all, dims=2)[:, 1] ./ sum(V_all, dims=2)[:, 1]
osc = sum(S_all, dims=2)[:, 1] ./ (35 * ρ₀)
mean_salinity = sum(S_all, dims=2)[:, 1] ./ sum(V_all, dims=2)[:, 1]

ohc_anomaly = ohc .- ohc[1]
osc_anomaly = osc .- osc[1]
ohc_anomaly_compare = ohc[target_overlap_idx] .- ohc[target_overlap_idx[1]]
osc_anomaly_compare = osc[target_overlap_idx] .- osc[target_overlap_idx[1]]
heat_flux_cumsum_compare .-= heat_flux_cumsum_compare[1]
fw_flux_cumsum_compare .-= fw_flux_cumsum_compare[1]

lines!(ax1, time_compare_years, ohc_anomaly_compare, label = "OHC anomaly")
lines!(ax1, time_compare_years, heat_flux_cumsum_compare, label = "cumsum(∫heat_flux dA · dt)")
lines!(ax2, time_in_years, mean_temperature .- mean_temperature[1], label = "Mean Temperature anomaly")
lines!(ax3, time_compare_years, osc_anomaly_compare, label = "OSC anomaly")
lines!(ax3, time_compare_years, fw_flux_cumsum_compare, label = "cumsum(∫fw_flux dA · dt)")
lines!(ax4, time_in_years, mean_salinity .- mean_salinity[1], label = "Mean Salinity anomaly")

ohc_difference = ohc_anomaly_compare .- heat_flux_cumsum_compare
osc_difference = osc_anomaly_compare .- fw_flux_cumsum_compare
lines!(ax5, time_compare_years, ohc_difference, label = "Difference")
lines!(ax6, time_compare_years, osc_difference, label = "Difference")
hlines!(ax5, [0.0], color = :black, linestyle = :dash)
hlines!(ax6, [0.0], color = :black, linestyle = :dash)

axislegend(ax1, position = :rb)
axislegend(ax3, position = :rb)

figpath_integrated = joinpath(figdir, "integrated_props_$(resolution).png")
save(figpath_integrated, fig, px_per_unit=3)

fig = Figure(size = (800, 500))
ax1 = Axis(fig[1, 1], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
ax2 = Axis(fig[2, 1], title = "Mean Temperature", xlabel = "Time (years)", ylabel = "Temperature (°C)")
ax3 = Axis(fig[1, 2], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
ax4 = Axis(fig[2, 2], title = "Mean Salinity", xlabel = "Time (years)", ylabel = "Salinity (psu)")

allranks_T_z_int = sum(T_z_all, dims=2)[:,1,:]
allranks_T_z_int_anomaly = allranks_T_z_int .- allranks_T_z_int[1,:]'
allranks_S_z_int = sum(S_z_all, dims=2)[:,1,:]
allranks_S_z_int_anomaly = allranks_S_z_int .- allranks_S_z_int[1,:]'
allranks_V_z_int = sum(V_z_all, dims=2)[:,1,:]
allranks_V_z_int_anomaly = allranks_V_z_int .- allranks_V_z_int[1,:]'

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
cleanup_copied_outputs!(copied_output_dir)
copied_output_dir = nothing

# fig = Figure(size = (800, 600))
# ax1 = Axis(fig[1, 1], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
# lines!(ax1, depth[1], 1035*1000*allranks_T_z_int_anomaly[16,:])
# xlims!(ax1, 0, -1000)
# ylims!(ax1, -1e20, 1e20)
# save(figdir * "test_temp_profile.png", fig, px_per_unit=3)
