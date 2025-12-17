using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2
using Glob
using OceanEnsembles

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved_fields/")
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
iterations = sort(collect(keys(iter_rank_map)))
ranks = iter_rank_map[iterations[1]]

time_total = []
T_int_iters = []
for iteration in iterations
    data = jldopen(string(unique_prefixes[1], "_iteration$(iteration)_rank", ranks[1], ".jld2"))
    timeiters = keys(data["timeseries/t/"])
    time = zeros(length(timeiters))
    for (i, iter) in enumerate(timeiters)
        time[i] = data["timeseries/t/$(iter)"]
    end
    push!(time_total, time)
    T_int = zeros(length(time), length(ranks))
    for rank in ranks
        data = jldopen(string(unique_prefixes[1], "_iteration$(iteration)_rank", rank, ".jld2"))
        for (i, iter) in enumerate(timeiters)
            T_int[i, rank+1] = data["timeseries/T_totintegral/$(iter)"][1,1,1]
        end
    end
    push!(T_int_iters, T_int)
end

T_all = vcat(T_int_iters...)
t_all = vcat(time_total...)
perm = sortperm(t_all)

t_sorted = t_all[perm]
T_sorted = T_all[perm, :]

time_in_years = t_sorted ./ (3600*24*365)


fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
lines!(ax1, time_in_years, 1035*1000*sum(T_sorted, dims=2)[:,1], label = "Ranked Integrals")
save(figdir * "OHC_$(resolution).png", fig, px_per_unit=3)
