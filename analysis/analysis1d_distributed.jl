using CairoMakie
using Oceananigans  # From local
using NumericalEarth: EarthSystemModels, Oceans
using Glob
using JLD2
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

resolution = "sxtdeg"
integral_files = glob("combined_global_tot_integrals_$(resolution)_RYF_run*.jld2", output_path)

run_number(file::AbstractString) = begin
    match_result = match(r"_run(\d+)\.jld2$", basename(file))
    isnothing(match_result) ? -1 : parse(Int, match_result.captures[1])
end

filter!(file -> run_number(file) >= 0, integral_files)
sort!(integral_files; by = run_number)
isempty(integral_files) && error("No combined integral run files found in $output_path")

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

function load_serialized_grid(file::AbstractString)
    return with_logger(NullLogger()) do
        jldopen(file, "r") do data
            read_serialized_grid(data)
        end
    end
end

time_total = Float64[]
T_total = Float64[]
S_total = Float64[]
V_total = Float64[]
T_z = Vector{Vector{Float64}}()
S_z = Vector{Vector{Float64}}()
V_z = Vector{Vector{Float64}}()

depth = Float64[]
integral_grid = load_serialized_grid(last(integral_files))
integral_grid === nothing && error("No serialized grid found in $(last(integral_files))")
integral_depth_grid = hasproperty(integral_grid, :underlying_grid) ? integral_grid.underlying_grid : integral_grid
append!(depth, integral_depth_grid.z.cᵃᵃᶜ)

records_by_time = Dict{Float64, NamedTuple{(:T, :S, :V, :Tz, :Sz, :Vz), Tuple{Float64, Float64, Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()

for file in integral_files
    @info "Reading integral diagnostics" file = basename(file)
    jldopen(file, "r") do data
        timeiters = sort(parse.(Int, keys(data["timeseries/t/"])))
        for iter in timeiters
            time = Float64(data["timeseries/t/$(iter)"])
            records_by_time[time] = (
                T = Float64(data["timeseries/T_totintegral/$(iter)"][1, 1, 1]),
                S = Float64(data["timeseries/S_totintegral/$(iter)"][1, 1, 1]),
                V = Float64(data["timeseries/total_volume_c/$(iter)"][1, 1, 1]),
                Tz = Float64.(vec(data["timeseries/T_vertintegral/$(iter)"][1, 1, 7:end-7])),
                Sz = Float64.(vec(data["timeseries/S_vertintegral/$(iter)"][1, 1, 7:end-7])),
                Vz = Float64.(vec(data["timeseries/vert_volume_c/$(iter)"][1, 1, 7:end-7])))
        end
    end
end

for time in sort!(collect(keys(records_by_time)))
    record = records_by_time[time]
    push!(time_total, time)
    push!(T_total, record.T)
    push!(S_total, record.S)
    push!(V_total, record.V)
    push!(T_z, record.Tz)
    push!(S_z, record.Sz)
    push!(V_z, record.Vz)
end

empty!(records_by_time)
isempty(time_total) && error("No timeseries entries found in $(join(basename.(integral_files), ", "))")
Lz = length(first(T_z))

if any(v -> length(v) != Lz, T_z) || any(v -> length(v) != Lz, S_z) || any(v -> length(v) != Lz, V_z)
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

T_z_matrix = permutedims(reduce(hcat, T_z))
S_z_matrix = permutedims(reduce(hcat, S_z))
V_z_matrix = permutedims(reduce(hcat, V_z))

@info "Loaded integral timeseries" steps = length(time_total)

empty!(T_z)
empty!(S_z)
empty!(V_z)
GC.gc()

t_all = time_total
time_in_years = t_all ./ (3600 * 24 * 365)

fig = Figure(size = (900, 600))
ax1 = Axis(fig[1, 1], title = "OHC Anomaly", xlabel = "Time (years)", ylabel = "Energy (J)")
ax2 = Axis(fig[1, 2], title = "Freshwater Content Anomaly", xlabel = "Time (years)", ylabel = "Freshwater mass (kg)")
ax3 = Axis(fig[2, 1], title = "Mean Temperature Anomaly", xlabel = "Time (years)", ylabel = "Temperature anomaly (°C)")
ax4 = Axis(fig[2, 2], title = "Mean Salinity Anomaly", xlabel = "Time (years)", ylabel = "Salinity anomaly (psu)")

ohc = ρ₀ * cₚ * T_total
mean_temperature = T_total ./ V_total
fw_content = ρ₀ .* (V_total .- S_total / 35)
mean_salinity = S_total ./ V_total

ohc_anomaly = ohc .- ohc[1]
fw_content_anomaly = fw_content .- fw_content[1]
mean_temperature_anomaly = mean_temperature .- mean_temperature[1]
mean_salinity_anomaly = mean_salinity .- mean_salinity[1]

lines!(ax1, time_in_years, ohc_anomaly)
lines!(ax2, time_in_years, fw_content_anomaly)
lines!(ax3, time_in_years, mean_temperature_anomaly)
lines!(ax4, time_in_years, mean_salinity_anomaly)


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
