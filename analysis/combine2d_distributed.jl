using Oceananigans
using Oceananigans.Fields: location
using JLD2
using Glob
using OceanEnsembles
using Dates

output_path = expanduser(get(ENV, "COMBINE_OUTPUT_PATH", "/home/tsohail/uom/ocean-ensembles/outputs/"))
figdir = expanduser(get(ENV, "COMBINE_FIGDIR", "/home/tsohail/uom/ocean-ensembles/figures/"))
rank_archive_path = joinpath(output_path, "rank_files")
mkpath(rank_archive_path)

resolution = get(ENV, "COMBINE_RESOLUTION", "sxtdeg")
nframes = nothing

combine_workers = parse(Int, get(ENV, "COMBINE_WORKERS", "1"))
combine_worker = parse(Int, get(ENV, "COMBINE_WORKER", "1"))
combine_gc_every = parse(Int, get(ENV, "COMBINE_GC_EVERY", "5"))

if combine_workers < 1
    error("COMBINE_WORKERS must be >= 1, got $(combine_workers)")
elseif combine_worker < 1 || combine_worker > combine_workers
    error("COMBINE_WORKER must be in 1:COMBINE_WORKERS, got $(combine_worker) of $(combine_workers)")
end

function shard_prefixes(prefixes, worker, workers)
    workers == 1 && return prefixes
    return [prefix for (idx, prefix) in enumerate(prefixes) if mod(idx - 1, workers) + 1 == worker]
end

function rss_string()
    status_path = "/proc/$(getpid())/status"
    isfile(status_path) || return "rss=?"

    for line in eachline(status_path)
        startswith(line, "VmRSS:") && return lowercase(strip(replace(line, r"\s+" => " ")))
    end

    return "rss=?"
end

function worker_log(message)
    println("[$(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")) worker $(combine_worker)/$(combine_workers) $(rss_string())] ", message)
    flush(stdout)
    return nothing
end

"""Move rank-split files for one run into outputs/rank_files."""
function archive_rank_files(prefix, ranks, run; iterrun = "run")
    worker_log("Archiving rank files for $(basename(prefix)) $(iterrun)$(run)")
    for rank in ranks
        source_path = prefix * "_$(iterrun)$(run)_rank$(rank).jld2"
        isfile(source_path) || continue

        destination_path = joinpath(rank_archive_path, basename(source_path))
        if isfile(destination_path)
            worker_log("  rank $(rank): duplicate already archived, removing source")
            @info "Rank file already archived at $(destination_path); removing duplicate source $(source_path)"
            rm(source_path; force = true)
        else
            worker_log("  rank $(rank): moving $(basename(source_path)) -> rank_files/")
            mv(source_path, destination_path)
        end
    end

    return nothing
end

"""Stitch one distributed variable into a single on-disk FieldTimeSeries efficiently."""
function set_distributed_field_time_series_fast!(fts, prefix, ranks, iteration, iters, grid; iterrun = "run")
    isempty(iters) && return nothing

    run = lpad(string(iteration), 4, '0')
    field = Field{location(fts)...}(grid)
    worker_log("Opening $(length(ranks)) rank files for $(basename(prefix)) $(iterrun)$(run), variable $(fts.name), $(length(iters)) frames")
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
        worker_log("  wrote frame 1/$(length(iters)) for $(fts.name)")

        for idx in 2:length(iters)
            iter = iters[idx]
            for rank in ranks
                data = rank_files[rank]["timeseries/$(fts.name)/$(iter)"][:, :, :]
                interior(field, :, y_slices[rank], :) .= data
            end

            set!(fts, field, idx)
            if idx == length(iters) || idx % 10 == 0
                worker_log("  wrote frame $(idx)/$(length(iters)) for $(fts.name)")
            end
            if combine_gc_every > 0 && idx % combine_gc_every == 0
                GC.gc(false)
            end
        end
    finally
        worker_log("Closing rank files for $(basename(prefix)) $(iterrun)$(run), variable $(fts.name)")
        foreach(close, values(rank_files))
        GC.gc(false)
    end

    return nothing
end

"""Combine all rank-split files for a single prefix using pre-indexed rank metadata."""
function combine_prefix_fast(prefix, grid, iter_rank_map; iterrun = "run")
    iterations = sort(collect(keys(iter_rank_map)))
    worker_log("Starting spatial prefix $(basename(prefix)) with $(length(iterations)) run(s)")

    for iteration in iterations
        run = lpad(string(iteration), 4, '0')
        ranks = sort(unique(iter_rank_map[iteration]))
        outpath = prefix * "_$(iterrun)$(run).jld2"

        if isfile(outpath)
            worker_log("Skipping $(basename(outpath)); combined output already exists")
            @info "Skipping run $run because output already exists: $outpath"
            archive_rank_files(prefix, ranks, run; iterrun)
            continue
        end

        file0_path = prefix * "_$(iterrun)$(run)_rank$(ranks[1]).jld2"
        worker_log("Inspecting $(basename(file0_path))")
        file0 = jldopen(file0_path, "r")

        if !haskey(file0, "timeseries/t")
            worker_log("Skipping $(basename(prefix)) $(iterrun)$(run): missing timeseries/t")
            @warn "Skipping run $run: key 'timeseries/t' not found in $file0_path"
            close(file0)
            continue
        end

        tkeys = collect(keys(file0["timeseries/t"]))
        iters = sort(parse.(Int, tkeys))
        times = Float64[file0["timeseries/t/$(k)"] for k in iters]
        vars = filter(!=("t"), collect(keys(file0["timeseries"])))

        @info "Combining ranks $(ranks) for run $(run)"
        worker_log("Combining $(basename(prefix)) $(iterrun)$(run): ranks=$(ranks), variables=$(length(vars)), frames=$(length(iters))")

        for var in vars
            @info "Writing $(var) for run $(run) → $outpath"
            worker_log("Writing variable $(var) -> $(basename(outpath))")
            rawlocs = file0["timeseries/$(var)/serialized/location"]
            fts = FieldTimeSeries{rawlocs[1], rawlocs[2], Nothing}(grid, times; backend = OnDisk(), path = outpath, name = string(var))
            set_distributed_field_time_series_fast!(fts, prefix, ranks, iteration, iters, grid; iterrun)
            hasproperty(fts, :output) && close(fts.output)
            fts = nothing
            GC.gc()
            worker_log("Finished variable $(var) -> $(basename(outpath))")
        end

        close(file0)
        archive_rank_files(prefix, ranks, run; iterrun)
        GC.gc()
        worker_log("Finished spatial prefix $(basename(prefix)) $(iterrun)$(run)")
    end

    return nothing
end

"""For total-integral outputs, keep the first-rank file as the combined output."""
function combine_prefix_first_rank(prefix, iter_rank_map; iterrun = "run")
    iterations = sort(collect(keys(iter_rank_map)))
    worker_log("Starting total-integral prefix $(basename(prefix)) with $(length(iterations)) run(s)")

    for iteration in iterations
        run = lpad(string(iteration), 4, '0')
        ranks = sort(unique(iter_rank_map[iteration]))
        outpath = prefix * "_$(iterrun)$(run).jld2"

        if isfile(outpath)
            worker_log("Skipping $(basename(outpath)); combined output already exists")
            @info "Skipping run $run because output already exists: $outpath"
            archive_rank_files(prefix, ranks, run; iterrun)
            continue
        end

        source_path = prefix * "_$(iterrun)$(run)_rank$(ranks[1]).jld2"
        if !isfile(source_path)
            worker_log("Skipping $(basename(prefix)) $(iterrun)$(run): source file not found")
            @warn "Skipping run $run: source file not found: $source_path"
            continue
        end

        @info "Copying first-rank totals file for run $(run): $(basename(source_path)) → $(basename(outpath))"
        worker_log("Copying totals $(basename(source_path)) -> $(basename(outpath))")
        cp(source_path, outpath)
        archive_rank_files(prefix, ranks, run; iterrun)
        worker_log("Finished total-integral prefix $(basename(prefix)) $(iterrun)$(run)")
    end

    return nothing
end

# Only include rank-split files so we don't parse already-combined outputs too.
files_fields = glob("global_*fields*$(resolution)*_RYF_run*_rank*.jld2", output_path)
files_fluxes = glob("global_surface_fluxes_$(resolution)*_RYF_run*_rank*.jld2", output_path)
files_totals = glob("global_*tot*$(resolution)*_RYF_run*_rank*.jld2", output_path)

files = vcat(files_fields, files_fluxes, files_totals)

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

is_total_prefix(prefix) = occursin("tot", basename(prefix))

spatial_prefixes = filter(!is_total_prefix, unique_prefixes)
total_prefixes = filter(is_total_prefix, unique_prefixes)

all_spatial_prefixes = spatial_prefixes
all_total_prefixes = total_prefixes
spatial_prefixes = shard_prefixes(all_spatial_prefixes, combine_worker, combine_workers)
total_prefixes = shard_prefixes(all_total_prefixes, combine_worker, combine_workers)

@info "Combine worker assignment" combine_worker combine_workers spatial_prefixes = length(spatial_prefixes) total_prefixes = length(total_prefixes)
worker_log("Assigned $(length(spatial_prefixes)) spatial prefix(es) and $(length(total_prefixes)) total prefix(es)")

if !isempty(spatial_prefixes)
    first_spatial_prefix = spatial_prefixes[1]
    first_iter_rank_map = prefix_iter_rank_map[first_spatial_prefix]
    iterations = sort(collect(keys(first_iter_rank_map)))
    ranks = sort(unique(first_iter_rank_map[iterations[1]]))
    run0 = lpad(string(iterations[1]), 4, '0')

    grid = create_grid(first_spatial_prefix * "_run$(run0)", ranks; gridtype = "TripolarGrid")
    @info "Grid created. Combining $(length(spatial_prefixes)) spatial prefixes."
    worker_log("Grid created from $(basename(first_spatial_prefix)) run$(run0); combining $(length(spatial_prefixes)) spatial prefix(es)")

    for prefix in spatial_prefixes
        @info "Combining spatial files for prefix: $prefix"
        worker_log("Queued spatial prefix $(basename(prefix))")
        combine_prefix_fast(prefix, grid, prefix_iter_rank_map[prefix])
    end
else
    @warn "No spatial rank-split files found for fields/fluxes. Skipping spatial combination."
end

if !isempty(total_prefixes)
    @info "Combining $(length(total_prefixes)) total-integral prefixes using first-rank files."
    for prefix in total_prefixes
        @info "Combining total-integral files for prefix: $prefix"
        worker_log("Queued total-integral prefix $(basename(prefix))")
        combine_prefix_first_rank(prefix, prefix_iter_rank_map[prefix])
    end
else
    @info "No total-integral rank-split files found."
end

worker_log("Done")
