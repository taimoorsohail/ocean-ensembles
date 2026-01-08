using Oceananigans  # From local
using JLD2
using Glob
using OceanEnsembles

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved_fields/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

resolution = "sxtdeg"
nframes = nothing
# Example: get all matching files in a folder
files = glob("global_*fields*$(resolution)*_RYF_iteration*.jld2", output_path)

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

@show iterations
@show ranks

grid = create_grid(unique_prefixes[1] * "_iteration$(iterations[1])", ranks; gridtype = "TripolarGrid")

@info "Grid created."

for prefix in unique_prefixes
    println("Combining files for prefix: $prefix")
    combine_ranks(prefix, grid)
end