#!/usr/bin/env julia

using Glob
using JLD2
using Oceananigans
using NumericalEarth
using Printf

const DEFAULT_PATTERN = "*run*.jld2"
const DEFAULT_OUTPUT_DIR = @__DIR__

with_trailing_slash(path) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator

@inline function run_id(path::AbstractString)
    m = match(r"_run(\d+)\.jld2$", basename(path))
    return m === nothing ? nothing : parse(Int, m.captures[1])
end

run_label(run::Integer) = @sprintf("%04d", run)

function print_usage()
    println("Usage: julia --project=ocean-ensembles/experiments/submission_scripts ocean-ensembles/outputs/combine_run_outputs.jl TARGET_RUN [GLOB ...] [--dry-run] [--output-dir PATH]")
    println()
    println("Examples:")
    println("  julia --project=ocean-ensembles/experiments/submission_scripts ocean-ensembles/outputs/combine_run_outputs.jl 1")
    println("  julia --project=ocean-ensembles/experiments/submission_scripts ocean-ensembles/outputs/combine_run_outputs.jl 1 'global_*_run*.jld2'")
    println("  julia --project=ocean-ensembles/experiments/submission_scripts ocean-ensembles/outputs/combine_run_outputs.jl 1 'global_16_fields_*_run*.jld2' --dry-run")
    return nothing
end

function parse_args(args::Vector{String})
    isempty(args) && return :help
    any(arg -> arg in ("-h", "--help"), args) && return :help

    dry_run = false
    output_dir = DEFAULT_OUTPUT_DIR
    positionals = String[]

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--dry-run"
            dry_run = true
            i += 1
        elseif arg == "--output-dir"
            i < length(args) || throw(ArgumentError("--output-dir requires a path argument."))
            output_dir = expanduser(args[i + 1])
            i += 2
        else
            push!(positionals, arg)
            i += 1
        end
    end

    isempty(positionals) && throw(ArgumentError("Missing TARGET_RUN argument."))
    target_run = tryparse(Int, first(positionals))
    target_run === nothing && throw(ArgumentError("Could not parse TARGET_RUN from '$(first(positionals))'."))
    target_run < 0 && throw(ArgumentError("TARGET_RUN must be non-negative, got $(target_run)."))

    patterns = length(positionals) > 1 ? positionals[2:end] : [DEFAULT_PATTERN]
    return (; target_run, patterns, dry_run, output_dir = with_trailing_slash(output_dir))
end

function collect_source_files(patterns::Vector{String}, output_dir::AbstractString)
    files = String[]
    seen = Set{String}()

    for pattern in patterns
        for file in glob(pattern, output_dir)
            abs = abspath(file)
            isfile(abs) || continue
            startswith(basename(abs), "combined_") && continue
            run = run_id(abs)
            isnothing(run) && continue
            abs in seen && continue
            push!(files, abs)
            push!(seen, abs)
        end
    end

    sort!(files; by = file -> (replace(basename(file), r"_run\d+\.jld2$" => ""), something(run_id(file), typemax(Int))))
    return files
end

function group_files_by_family(files::Vector{String})
    grouped = Dict{String, Vector{String}}()

    for file in files
        family = replace(basename(file), r"_run\d+\.jld2$" => "")
        push!(get!(grouped, family, String[]), file)
    end

    for group in values(grouped)
        sort!(group; by = file -> something(run_id(file), typemax(Int)))
    end

    return grouped
end

function list_timeseries_variables(file)
    vars = String[]
    for key in keys(file["timeseries"])
        name = String(key)
        name == "t" && continue
        push!(vars, name)
    end
    sort!(vars)
    return vars
end

function validate_timeseries_file(path::AbstractString)
    jldopen(path, "r") do f
        haskey(f, "timeseries") || return false, "missing timeseries group"
        haskey(f, "timeseries/t") || return false, "missing timeseries/t"
        return true, nothing
    end
end


function recursive_copy_group!(dst, src_group, prefix::AbstractString)
    for key in keys(src_group)
        name = String(key)
        child_path = isempty(prefix) ? name : string(prefix, "/", name)
        child = src_group[name]
        if child isa JLD2.Group
            recursive_copy_group!(dst, child, child_path)
        else
            dst[child_path] = child
        end
    end
    return nothing
end

function copy_non_timeseries_metadata!(dst, src)
    for key in keys(src)
        name = String(key)
        name == "timeseries" && continue
        child = src[name]
        if child isa JLD2.Group
            recursive_copy_group!(dst, child, name)
        else
            dst[name] = child
        end
    end
    return nothing
end

function copy_timeseries_schema!(dst, schema_sources::Dict{String, String}, vars::Vector{String})
    for var in vars
        source_path = get(schema_sources, var, nothing)
        isnothing(source_path) && continue
        jldopen(source_path, "r") do src
            if haskey(src["timeseries/$var"], "serialized")
                recursive_copy_group!(dst, src["timeseries/$var/serialized"], "timeseries/$var/serialized")
            end
        end
    end
    return nothing
end

function ensure_serialized_grid_aliases!(dst, representative_file::AbstractString, vars::Vector{String})
    needed_indices = Int[]

    for var in vars
        grid_index_path = "timeseries/$var/serialized/grid_index"
        haskey(dst, grid_index_path) || continue
        idx = Int(dst[grid_index_path])
        idx in needed_indices || push!(needed_indices, idx)
    end

    isempty(needed_indices) && return nothing
    haskey(dst, "serialized/grid") || return nothing

    jldopen(representative_file, "r") do rep
        source_grid = haskey(rep, "serialized/grid") ? rep["serialized/grid"] : dst["serialized/grid"]
        for idx in needed_indices
            alias_path = "serialized/grid_$(idx)"
            haskey(dst, alias_path) && continue
            dst[alias_path] = source_grid
        end
    end

    return nothing
end

function nan_template_for_var(file, var::String)
    keys_for_var = sort(parse.(Int, filter(!=("serialized"), string.(collect(keys(file["timeseries/$var"]))))))
    isempty(keys_for_var) && error("No data keys found for variable $var when building NaN template.")
    raw = file["timeseries/$var/$(first(keys_for_var))"]
    src = raw isa AbstractArray && ndims(raw) == 3 ? raw[:, :, :] : raw isa AbstractArray && ndims(raw) == 2 ? raw[:, :] : nothing
    src === nothing && error("Unsupported array rank for variable $var when building NaN template.")
    template = Array{Float32}(undef, size(src)...)
    fill!(template, NaN32)
    return template
end

function build_selected_records(files::Vector{String})
    selected = Dict{Float64, NamedTuple{(:run, :path, :source_key), Tuple{Int, String, String}}}()
    replaced_duplicates = 0
    var_set = Set{String}()
    schema_sources = Dict{String, String}()
    usable_files = String[]
    skipped_files = Pair{String, String}[]

    for file in files
        ok, reason = validate_timeseries_file(file)
        if !ok
            push!(skipped_files, file => something(reason, "unrecognized file layout"))
            continue
        end

        push!(usable_files, file)
        run = run_id(file)
        isnothing(run) && error("Missing run id for $(file).")
        run = run::Int
        jldopen(file, "r") do f
            vars = list_timeseries_variables(f)
            for var in vars
                push!(var_set, var)
                get!(schema_sources, var, file)
            end

            time_keys = sort(parse.(Int, string.(collect(keys(f["timeseries/t"])))) )
            for key in time_keys
                tval = Float64(f["timeseries/t/$(key)"])
                existing = get(selected, tval, nothing)
                if isnothing(existing) || run >= existing.run
                    replaced_duplicates += !isnothing(existing) && run > existing.run ? 1 : 0
                    selected[tval] = (run = run, path = file, source_key = string(key))
                end
            end
        end
    end

    isempty(usable_files) && error("No readable timeseries files were found in this family.")
    isempty(var_set) && error("No readable timeseries variables were found.")
    vars = sort!(collect(var_set))
    sorted_times = sort!(collect(keys(selected)))
    return sorted_times, selected, vars, replaced_duplicates, schema_sources, usable_files, skipped_files
end

function target_output_path(representative_file::AbstractString, target_run::Int)
    base = replace(basename(representative_file), r"_run\d+\.jld2$" => "")
    return joinpath(dirname(representative_file), string("combined_", base, "_run", run_label(target_run), ".jld2"))
end

function write_merged_file!(target_path::AbstractString,
                            representative_file::AbstractString,
                            sorted_times::Vector{Float64},
                            selected_records::Dict{Float64, NamedTuple{(:run, :path, :source_key), Tuple{Int, String, String}}},
                            vars::Vector{String},
                            schema_sources::Dict{String, String})
    temp_path = target_path * ".tmp"
    isfile(temp_path) && rm(temp_path; force = true)

    jldopen(representative_file, "r") do rep
        jldopen(temp_path, "w") do out
            copy_non_timeseries_metadata!(out, rep)
            copy_timeseries_schema!(out, schema_sources, vars)
            ensure_serialized_grid_aliases!(out, representative_file, vars)

            nan_templates = Dict{String, Array{Float32}}()
            for var in vars
                source_path = get(schema_sources, var, nothing)
                isnothing(source_path) && continue
                jldopen(source_path, "r") do schema_file
                    nan_templates[var] = nan_template_for_var(schema_file, var)
                end
            end

            active_path = nothing
            active_file = nothing

            try
                for (index, tval) in enumerate(sorted_times)
                    record = selected_records[tval]
                    if record.path != active_path
                        if active_file !== nothing
                            close(active_file)
                        end
                        active_file = jldopen(record.path, "r")
                        active_path = record.path
                    end

                    out_key = string(index - 1)
                    out["timeseries/t/$out_key"] = tval
                    for var in vars
                        source_dataset = "timeseries/$var/$(record.source_key)"
                        if haskey(active_file, source_dataset)
                            out["timeseries/$var/$out_key"] = active_file[source_dataset]
                        else
                            out["timeseries/$var/$out_key"] = nan_templates[var]
                        end
                    end
                end
            finally
                if active_file !== nothing
                    close(active_file)
                end
            end
        end
    end

    mv(temp_path, target_path; force = true)
    return nothing
end

function merge_family!(files::Vector{String}, target_run::Int; dry_run::Bool)
    sorted_times, selected_records, vars, replaced_duplicates, schema_sources, usable_files, skipped_files = build_selected_records(files)
    representative = first(usable_files)
    target_path = target_output_path(representative, target_run)

    println("Family: $(replace(basename(representative), r"_run\d+\.jld2$" => ""))")
    println("  Source runs: $(join(run_label.(something.(run_id.(usable_files), 0)), ", "))")
    println("  Frames after merge: $(length(sorted_times))")
    println("  Variables in merged output: $(join(vars, ", "))")
    println("  Duplicate times replaced by later runs: $(replaced_duplicates)")
    if !isempty(skipped_files)
        println("  Skipped files: $(length(skipped_files))")
        for (file, reason) in skipped_files
            println("    - $(basename(file)): $(reason)")
        end
    end
    println("  Output: $(target_path)")

    if dry_run
        return nothing
    end

    if isfile(target_path)
        @warn "Skipping existing combined output." target_path
        return nothing
    end

    write_merged_file!(target_path, representative, sorted_times, selected_records, vars, schema_sources)
    return nothing
end

function main(args::Vector{String})
    parsed = parse_args(args)
    parsed === :help && return print_usage()

    files = collect_source_files(parsed.patterns, parsed.output_dir)
    isempty(files) && error("No matching run files were found in $(parsed.output_dir) for patterns $(parsed.patterns).")

    grouped = group_files_by_family(files)
    families = sort!(collect(keys(grouped)))

    println("Scanning $(length(files)) files across $(length(families)) output families in $(parsed.output_dir)")
    println("Target run label: run$(run_label(parsed.target_run))")
    parsed.dry_run && println("Dry-run mode: no files will be written.")

    for family in families
        merge_family!(grouped[family], parsed.target_run; dry_run = parsed.dry_run)
    end

    return nothing
end

try
    main(ARGS)
catch err
    if err isa InterruptException
        rethrow()
    end
    showerror(stderr, err)
    println(stderr)
    print_usage()
    exit(1)
end
