using CairoMakie
using Glob
using JLD2
using Oceananigans
using Statistics

const OUTPUT_PATH = normpath(joinpath(@__DIR__, "..", "outputs"))
const FIG_DIR = normpath(joinpath(@__DIR__, "..", "figures"))
const RESOLUTION = "sxtdeg"
const SECONDS_PER_YEAR = 365 * 24 * 60 * 60
const WATTS_PER_PETAWATT = 1e15
const TARGET_LATITUDE = 26.0
const TRENBERTH_MHT_PATH = joinpath(@__DIR__, "..", "data", "Trenberth_et_al_2019",
                                    "Fig_4a_2000_2016_monthly_ZonalSum_MeanAnnualCycle_globe_0-2000m_ensemble_mean.txt")

run_number(path::AbstractString) = begin
    match_result = match(r"_run(\d+)\.jld2$", basename(path))
    isnothing(match_result) ? -1 : parse(Int, only(match_result.captures))
end

function latest_mht_file()
    pattern = "combined_global_MHT_$(RESOLUTION)_RYF_run*.jld2"
    files = filter(file -> run_number(file) >= 0, glob(pattern, OUTPUT_PATH))
    isempty(files) && error("No files matching $pattern found in $OUTPUT_PATH.")
    return last(sort(files; by = run_number))
end

function load_mht(path::AbstractString)
    return jldopen(path, "r") do file
        haskey(file, "timeseries/mht") || error("timeseries/mht not found in $path.")
        haskey(file, "timeseries/t") || error("timeseries/t not found in $path.")
        haskey(file, "serialized/grid") || error("serialized/grid not found in $path.")

        data_keys = filter(!=("serialized"), String.(collect(keys(file["timeseries/mht"]))))
        iterations = sort(parse.(Int, data_keys))
        isempty(iterations) && error("No MHT frames found in $path.")

        times = Float64[file["timeseries/t/$iteration"] for iteration in iterations]
        first_frame = file["timeseries/mht/$(first(iterations))"]
        nlat = length(first_frame)
        mht = Matrix{Float64}(undef, length(iterations), nlat)

        for (frame, iteration) in enumerate(iterations)
            values = vec(file["timeseries/mht/$iteration"])
            length(values) == nlat || error("MHT frame $iteration has an inconsistent size.")
            mht[frame, :] .= values
        end

        grid = file["serialized/grid"]
        latitude_matrix = φnodes(grid, Center(), Face(), Center())
        latitudes = vec(mean(latitude_matrix; dims = 1))
        length(latitudes) == nlat || error("Latitude and MHT sizes do not match.")

        order = sortperm(times)
        return times[order], latitudes, mht[order, :]
    end
end

function time_mean(values::AbstractMatrix, times::AbstractVector)
    size(values, 1) == length(times) || error("MHT and time dimensions do not match.")
    length(times) == 1 && return vec(values[1, :])

    means = fill(NaN, size(values, 2))
    for j in axes(values, 2)
        weighted_sum = 0.0
        total_time = 0.0
        for i in 1:length(times)-1
            left, right = values[i, j], values[i + 1, j]
            if isfinite(left) && isfinite(right)
                Δt = times[i + 1] - times[i]
                Δt > 0 || continue
                weighted_sum += (left + right) * Δt / 2
                total_time += Δt
            end
        end
        total_time > 0 && (means[j] = weighted_sum / total_time)
    end
    return means
end

function load_trenberth_global_mht(path::AbstractString)
    isfile(path) || error("Trenberth et al. (2019) MHT data not found: $path")

    latitudes = Float64[]
    annual_mean_mht = Float64[]
    for line in eachline(path)
        columns = split(strip(line))
        length(columns) == 14 || continue
        values = tryparse.(Float64, columns)
        any(isnothing, values) && continue
        push!(latitudes, something(first(values), NaN))
        push!(annual_mean_mht, something(last(values), NaN))
    end

    isempty(latitudes) && error("No global MHT values could be read from $path.")
    order = sortperm(latitudes)
    return latitudes[order], annual_mean_mht[order]
end

function symmetric_color_range(values)
    finite_values = filter(isfinite, vec(values))
    isempty(finite_values) && error("MHT contains no finite values.")
    limit = quantile(abs.(finite_values), 0.99)
    limit == 0 && (limit = maximum(abs, finite_values))
    limit == 0 && (limit = 1.0)
    return (-limit, limit)
end

input_file = isempty(ARGS) ? latest_mht_file() : abspath(expanduser(first(ARGS)))
isfile(input_file) || error("MHT input file does not exist: $input_file")
mkpath(FIG_DIR)

@info "Loading MHT analysis input." input_file
times, latitudes, mht_watts = load_mht(input_file)
years = times ./ SECONDS_PER_YEAR
mht_pw = mht_watts ./ WATTS_PER_PETAWATT

# Figure 1: Hovmöller diagram.
fig1 = Figure(size = (1100, 700))
ax1 = Axis(fig1[1, 1];
           title = "Meridional Heat Transport",
           xlabel = "Year",
           ylabel = "Nominal latitude (°N)")
hm = heatmap!(ax1, years, latitudes, mht_pw;
              colormap = :vik,
              colorrange = symmetric_color_range(mht_pw))
Colorbar(fig1[1, 2], hm; label = "MHT (PW)")
fig1_path = joinpath(FIG_DIR, "mht_hovmoller_$(RESOLUTION).png")
save(fig1_path, fig1; px_per_unit = 3)

# Figure 2: time-mean MHT.
mean_mht_pw = time_mean(mht_pw, times)
trenberth_latitudes, trenberth_mean_mht_pw = load_trenberth_global_mht(TRENBERTH_MHT_PATH)
fig2 = Figure(size = (900, 650))
ax2 = Axis(fig2[1, 1];
           title = "Time-mean Meridional Heat Transport",
           xlabel = "Nominal latitude (°N)",
           ylabel = "MHT (PW)")
lines!(ax2, latitudes, mean_mht_pw; linewidth = 3, label = "OceanEnsembles")
lines!(ax2, trenberth_latitudes, trenberth_mean_mht_pw;
       linewidth = 3,
       linestyle = :dash,
       color = :black,
       label = "Trenberth et al., 2019")
hlines!(ax2, 0; color = (:black, 0.4), linestyle = :dash)
axislegend(ax2; position = :rt)
fig2_path = joinpath(FIG_DIR, "mht_time_mean_$(RESOLUTION).png")
save(fig2_path, fig2; px_per_unit = 3)

# Figure 3: MHT at the model row nearest 26°N.
latitude_index = argmin(abs.(latitudes .- TARGET_LATITUDE))
selected_latitude = latitudes[latitude_index]
fig3 = Figure(size = (1000, 600))
ax3 = Axis(fig3[1, 1];
           title = "Meridional Heat Transport at $(round(selected_latitude; digits = 2))°N",
           xlabel = "Year",
           ylabel = "MHT (PW)")
lines!(ax3, years, view(mht_pw, :, latitude_index); linewidth = 2)
hlines!(ax3, 0; color = (:black, 0.4), linestyle = :dash)
fig3_path = joinpath(FIG_DIR, "mht_26N_$(RESOLUTION).png")
save(fig3_path, fig3; px_per_unit = 3)

@info "Saved MHT figures." fig1_path fig2_path fig3_path selected_latitude
