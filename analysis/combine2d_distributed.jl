using Oceananigans
using Oceananigans.Fields: location
using JLD2
using Glob
using OceanEnsembles

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

resolution = "sxtdeg"
nframes = nothing

"""Stitch one distributed variable into a single on-disk FieldTimeSeries efficiently."""
function set_distributed_field_time_series_fast!(fts, prefix, ranks, iteration, iters, grid; iterrun = "run")
    isempty(iters) && return nothing

    run = lpad(string(iteration), 4, '0')
    field = Field{location(fts)...}(grid)
    rank_files = Dict(rank => jldopen(prefix * "_$(iterrun)$(run)_rank$(rank).jld2", "r") for rank in ranks)

    try
        y_slices = Dict{Int, UnitRange{Int}}()
        y_start = 1
        first_iter = iters[1]

        for rank in ranks
            data = rank_files[rank]["timeseries/$(fts.name)/$(first_iter)"][:, :, :]
            y_stop = y_start + size(data, 2) - 1
            y_slice = y_start:y_stop
            y_slices[rank] = y_slice
            interior(field, :, y_slice, :) .= data
            y_start = y_stop + 1
        end

        set!(fts, field, 1)

        for idx in 2:length(iters)
            iter = iters[idx]
            for rank in ranks
                data = rank_files[rank]["timeseries/$(fts.name)/$(iter)"][:, :, :]
                interior(field, :, y_slices[rank], :) .= data
            end

            set!(fts, field, idx)
        end
    finally
        foreach(close, values(rank_files))
    end

    return nothing
end

"""Combine all rank-split files for a single prefix using pre-indexed rank metadata."""
function combine_prefix_fast(prefix, grid, iter_rank_map; iterrun = "run")
    iterations = sort(collect(keys(iter_rank_map)))

    for iteration in iterations
        run = lpad(string(iteration), 4, '0')
        ranks = sort(unique(iter_rank_map[iteration]))
        outpath = prefix * "_$(iterrun)$(run).jld2"

        if isfile(outpath)
            @info "Skipping run $run because output already exists: $outpath"
            continue
        end

        file0_path = prefix * "_$(iterrun)$(run)_rank$(ranks[1]).jld2"
        file0 = jldopen(file0_path, "r")

        if !haskey(file0, "timeseries/t")
            @warn "Skipping run $run: key 'timeseries/t' not found in $file0_path"
            close(file0)
            continue
        end

        tkeys = collect(keys(file0["timeseries/t"]))
        iters = sort(parse.(Int, tkeys))
        times = Float64[file0["timeseries/t/$(k)"] for k in iters]
        vars = filter(!=("t"), collect(keys(file0["timeseries"])))

        @info "Combining ranks $(ranks) for run $(run)"

        for var in vars
            @info "Writing $(var) for run $(run) → $outpath"
            rawlocs = file0["timeseries/$(var)/serialized/location"]
            fts = FieldTimeSeries{rawlocs[1], rawlocs[2], Nothing}(grid, times; backend = OnDisk(), path = outpath, name = string(var))
            set_distributed_field_time_series_fast!(fts, prefix, ranks, iteration, iters, grid; iterrun)
            hasproperty(fts, :output) && close(fts.output)
        end

        close(file0)
        GC.gc()
    end

    return nothing
end

# Only include rank-split files so we don't parse already-combined outputs too.
files = glob("global_*fields*$(resolution)*_RYF_run*_rank*.jld2", output_path)

if isempty(files)
    error("No rank-split files found for resolution=$(resolution) in $(output_path)")
end

# Build prefix => iteration => ranks in one pass, avoiding repeated directory scans.
prefix_iter_rank_map = Dict{String, Dict{Int, Vector{Int}}}()
pattern = r"^(.*)_run(\d+)_rank(\d+)\.jld2$"

for file in files
    fname = basename(file)
    m = match(pattern, fname)
    m === nothing && continue

    prefix = joinpath(output_path, m.captures[1])
    iteration = parse(Int, m.captures[2])
    rank = parse(Int, m.captures[3])

    local iter_rank_map = get!(prefix_iter_rank_map, prefix, Dict{Int, Vector{Int}}())
    push!(get!(iter_rank_map, iteration, Int[]), rank)
end

unique_prefixes = sort(collect(keys(prefix_iter_rank_map)))

if isempty(unique_prefixes)
    error("No valid prefix/run/rank filenames found in $(output_path)")
end

first_prefix = unique_prefixes[1]
first_iter_rank_map = prefix_iter_rank_map[first_prefix]
iterations = sort(collect(keys(first_iter_rank_map)))
ranks = sort(unique(first_iter_rank_map[iterations[1]]))
run0 = lpad(string(iterations[1]), 4, '0')

grid = create_grid(first_prefix * "_run$(run0)", ranks; gridtype = "TripolarGrid")

@info "Grid created. Combining $(length(unique_prefixes)) prefixes."

for prefix in unique_prefixes
    @info "Combining files for prefix: $prefix"
    combine_prefix_fast(prefix, grid, prefix_iter_rank_map[prefix])
end
