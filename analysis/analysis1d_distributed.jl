using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2
using Glob
using OceanEnsembles

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

resolution = "sxtdeg"
nframes = nothing
# Example: get all matching files in a folder
files = glob("global_*tot*$(resolution)*_RYF_iteration*.jld2", output_path)

# Collect prefixes here
prefixes = String[]

for file in files
    fname = basename(file)
    prefix = replace(fname, r"_iteration.*" => "")
    push!(prefixes, joinpath(output_path, prefix))
end


# Keep only unique prefixes
unique_prefixes = unique(prefixes)

println(unique_prefixes)

iter_rank_map = identify_combination_targets(basename(unique_prefixes[1]), dirname(unique_prefixes[1]); type = "iterrank")
iterations = collect(keys(iter_rank_map))
ranks = iter_rank_map[iterations[1]]

data = jldopen(string(unique_prefixes[1], "_iteration", iterations[1], "_rank", ranks[1], ".jld2"))
timeiters = keys(data["timeseries/t/"])
time= zeros(length(timeiters))
for (i, iter) in enumerate(timeiters)
    time[i] = data["timeseries/t/$(iter)"]
end
time_in_years = time ./ (3600*24*365)
T_int = zeros(length(time), length(ranks))
for rank in ranks
    data = jldopen(string(unique_prefixes[1], "_iteration", iterations[1], "_rank", rank, ".jld2"))
    for (i, iter) in enumerate(timeiters)
        T_int[i, rank+1] = data["timeseries/T_totintegral/$(iter)"][1,1,1]
    end
end

fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], title = "Total Temperature Integral per Rank", xlabel = "Time (years)", ylabel = "OHC (J)")
lines!(ax1, time_in_years, 1035*1000*sum(T_int, dims=2)[:,1], label = "Ranked Integrals")
save(figdir * "OHC_$(resolution).png", fig, px_per_unit=3)
