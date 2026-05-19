using CairoMakie
using JLD2
using Glob
using Statistics
using Printf

with_trailing_slash(path) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator

const OUTPUT_PATH = with_trailing_slash(expanduser(get(ENV, "OUTPUT_PATH", "/home/tsohail/uom/ocean-ensembles/outputs/")))
const FIGDIR = with_trailing_slash(expanduser(get(ENV, "FIGDIR", "/home/tsohail/uom/ocean-ensembles/figures/")))
const ANALYSIS_OUTPUT_PATH = with_trailing_slash(expanduser(get(ENV, "COMPARE_OUTPUT_PATH", FIGDIR)))
const RESOLUTION = "sxtdeg"
const FIGURE_FORMAT = get(ENV, "COMPARE_FIGURE_FORMAT", "png")
const TARGET_DEPTH = parse(Int, get(ENV, "COMPARE_DEPTH_LEVEL", "75"))

const VAR_TITLES = Dict(
    "T" => "Temperature",
    "S" => "Salinity",
    "u" => "Zonal Velocity",
    "v" => "Meridional Velocity",
    "w" => "Vertical Velocity",
    "e" => "Tracer e"
)

function usage()
    println("Usage: julia --project=ocean-ensembles/analysis ocean-ensembles/analysis/compare_timeaveragedinterval_manual_average.jl [run_id] [depth_level]")
    println("If run_id is omitted, the latest run with both averaged and instantaneous files is used.")
    println("If depth_level is omitted, COMPARE_DEPTH_LEVEL or 75 is used.")
end

@inline function run_id(path::AbstractString)
    m = match(r"run(\d+)", basename(path))
    return m === nothing ? -1 : parse(Int, m.captures[1])
end

@inline function depth_level(path::AbstractString)
    m = match(r"global_(\d+)_fields_", basename(path))
    return m === nothing ? -1 : parse(Int, m.captures[1])
end

@inline function extract_2d_f32(raw)
    if ndims(raw) == 2
        return Float32.(raw)
    elseif ndims(raw) == 3
        return Float32.(raw[:, :, 1])
    end

    return nothing
end

numeric_timeseries_keys(group) = sort(parse.(Int, collect(keys(group))))
variable_title(var::String) = get(VAR_TITLES, var, var)

function averaged_depth_files(path::AbstractString)
    files = glob("global_*_fields_$(RESOLUTION)_RYF_run*.jld2", path)
    files = filter(files) do f
        !occursin("_rank", f) &&
        !occursin("_instantaneous_", basename(f)) &&
        occursin(r"global_\d+_fields_", basename(f)) &&
        run_id(f) >= 0
    end
    sort!(files; by = f -> (run_id(f), depth_level(f)))
    return files
end

function instantaneous_depth_files(path::AbstractString)
    files = glob("global_*_fields_$(RESOLUTION)_RYF_instantaneous_run*.jld2", path)
    files = filter(files) do f
        !occursin("_rank", f) &&
        occursin(r"global_\d+_fields_", basename(f)) &&
        run_id(f) >= 0
    end
    sort!(files; by = f -> (run_id(f), depth_level(f)))
    return files
end

function choose_run_id(path::AbstractString)
    averaged_runs = Set(run_id.(averaged_depth_files(path)))
    instantaneous_runs = Set(run_id.(instantaneous_depth_files(path)))
    common_runs = sort!(collect(intersect(averaged_runs, instantaneous_runs)))
    isempty(common_runs) && error("No run IDs found with both averaged and instantaneous depth files in $path")
    return common_runs[end]
end

function files_by_depth(files::Vector{String}, selected_run::Int)
    mapping = Dict{Int, String}()
    for file in files
        run_id(file) == selected_run || continue
        depth = depth_level(file)
        depth >= 0 || continue
        mapping[depth] = file
    end
    return mapping
end

function timeseries_variable_names(filepath::AbstractString)
    return jldopen(filepath, "r") do f
        haskey(f, "timeseries") || error("Missing timeseries group in $filepath")
        names = String[]
        for key in keys(f["timeseries"])
            key == "t" && continue
            push!(names, String(key))
        end
        sort!(names)
        return names
    end
end

function load_all_frames(filepath::AbstractString, var::String)
    return jldopen(filepath, "r") do f
        haskey(f, "timeseries/t") || error("Missing timeseries/t in $filepath")
        haskey(f, "timeseries/$var") || error("Missing timeseries/$var in $filepath")

        keys_t = numeric_timeseries_keys(f["timeseries/t"])
        times = Float64[]
        frames = Matrix{Float32}[]

        for key in keys_t
            raw = extract_2d_f32(f["timeseries/$var/$key"])
            raw === nothing && continue
            push!(times, Float64(f["timeseries/t/$key"]))
            push!(frames, raw)
        end

        isempty(frames) && error("No readable frames found for $var in $filepath")
        return times, frames
    end
end

function load_single_averaged_frame(filepath::AbstractString, var::String)
    return jldopen(filepath, "r") do f
        haskey(f, "timeseries/t") || error("Missing timeseries/t in $filepath")
        haskey(f, "timeseries/$var") || error("Missing timeseries/$var in $filepath")

        keys_t = numeric_timeseries_keys(f["timeseries/t"])
        isempty(keys_t) && error("No averaged times found in $filepath")

        if length(keys_t) > 1
            @warn "Multiple averaged snapshots found; using the last one." filepath variable = var count = length(keys_t)
        end

        key = keys_t[end]
        frame = extract_2d_f32(f["timeseries/$var/$key"])
        frame === nothing && error("Could not read averaged frame for $var in $filepath")
        return Float64(f["timeseries/t/$key"]), frame, length(keys_t)
    end
end

function arithmetic_mean(frames::Vector{Matrix{Float32}})
    accumulator = zeros(Float64, size(frames[1]))
    for frame in frames
        @inbounds accumulator .+= frame
    end
    accumulator ./= length(frames)
    return accumulator
end

rms(A) = sqrt(mean(abs2, A))

function compare_depth_variable(instantaneous_file::String, averaged_file::String, var::String)
    inst_times, inst_frames = load_all_frames(instantaneous_file, var)
    averaged_time, averaged_frame, averaged_snapshot_count = load_single_averaged_frame(averaged_file, var)

    averaging_start_index = !isempty(inst_times) && iszero(first(inst_times)) ? 2 : 1
    averaging_times = inst_times[averaging_start_index:end]
    averaging_frames = inst_frames[averaging_start_index:end]
    isempty(averaging_frames) && error("No instantaneous frames remain after excluding the t=0 snapshot for $var")

    manual_average = arithmetic_mean(averaging_frames)
    averaged_float = Float64.(averaged_frame)
    signed_error = manual_average .- averaged_float
    abs_error = abs.(signed_error)

    reference_rms = rms(averaged_float)
    relative_rms_error = iszero(reference_rms) ? NaN : rms(signed_error) / reference_rms

    summary = (
        sample_count = length(averaging_frames),
        source_file_count = 1,
        averaged_snapshot_count = averaged_snapshot_count,
        start_time = first(averaging_times),
        end_time = last(averaging_times),
        averaged_time = averaged_time,
        mean_manual = mean(manual_average),
        mean_averaged = mean(averaged_float),
        bias = mean(signed_error),
        mean_abs_error = mean(abs_error),
        rms_error = rms(signed_error),
        max_abs_error = maximum(abs_error),
        relative_rms_error = relative_rms_error
    )

    payload = (
        instantaneous_times = averaging_times,
        manual_average = Float32.(manual_average),
        averaged = averaged_frame,
        signed_error = Float32.(signed_error),
        abs_error = Float32.(abs_error)
    )

    return summary, payload
end

function write_summary_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "depth,variable,sample_count,source_file_count,averaged_snapshot_count,start_time,end_time,averaged_time,mean_manual,mean_averaged,bias,mean_abs_error,rms_error,max_abs_error,relative_rms_error")
        for row in rows
            @printf(io, "%d,%s,%d,%d,%d,%.8f,%.8f,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                    row.depth, row.variable, row.sample_count, row.source_file_count, row.averaged_snapshot_count,
                    row.start_time, row.end_time, row.averaged_time,
                    row.mean_manual, row.mean_averaged, row.bias, row.mean_abs_error,
                    row.rms_error, row.max_abs_error, row.relative_rms_error)
        end
    end

    return nothing
end

function print_summary(rows)
    println()
    println("Depth  Var  Samples  Files  RMS Error    Max Abs Error    Rel RMS Error")
    println("-----  ---  -------  -----  -----------  ---------------  ---------------")
    for row in rows
        @printf("%5d  %-3s  %7d  %5d  %11.6e  %15.6e  %15.6e\n",
                row.depth, row.variable, row.sample_count, row.source_file_count,
                row.rms_error, row.max_abs_error, row.relative_rms_error)
    end
    println()
end

function field_colorrange(A)
    finite_values = filter(isfinite, vec(Float64.(A)))
    isempty(finite_values) && return (-1.0, 1.0)
    lo = minimum(finite_values)
    hi = maximum(finite_values)
    lo == hi && return (lo - 1e-12, hi + 1e-12)
    return (lo, hi)
end

function error_colorrange(A)
    finite_values = filter(isfinite, vec(Float64.(A)))
    isempty(finite_values) && return (-1.0, 1.0)
    limit = maximum(abs, finite_values)
    iszero(limit) && (limit = 1e-12)
    return (-limit, limit)
end

function save_surface_comparison_figure(compare_vars::Vector{String}, results_for_depth::Dict{String, Any}, depth::Int, run_label::String)
    nvars = length(compare_vars)
    fig = Figure(size = (1700, max(280 * nvars, 420)))

    sample_counts = [results_for_depth[var].summary.sample_count for var in compare_vars]
    file_counts = [results_for_depth[var].summary.source_file_count for var in compare_vars]
    Label(fig[0, 1:5],
          "Depth level $(depth): AveragedTimeInterval vs manual instantaneous average (run $(run_label), $(first(sample_counts)) samples from $(first(file_counts)) file)",
          fontsize = 24,
          tellwidth = false)

    column_titles = ("Averaged", "Manual Average", "Error")
    for (col, title) in enumerate(column_titles)
        Label(fig[1, col], title, fontsize = 18, tellwidth = false)
    end

    for (row_index, var) in enumerate(compare_vars)
        payload = results_for_depth[var]
        plot_row = row_index + 1

        averaged = payload.averaged
        manual = payload.manual_average
        signed_error = payload.signed_error

        field_limits = field_colorrange(vcat(vec(Float64.(averaged)), vec(Float64.(manual))))
        error_limits = error_colorrange(signed_error)

        ax_avg = Axis(fig[plot_row, 1], ylabel = variable_title(var), xlabel = "i", aspect = DataAspect())
        ax_manual = Axis(fig[plot_row, 2], xlabel = "i", aspect = DataAspect())
        ax_error = Axis(fig[plot_row, 3], xlabel = "i", aspect = DataAspect())

        hm_avg = heatmap!(ax_avg, averaged; colormap = :thermal, colorrange = field_limits)
        hm_manual = heatmap!(ax_manual, manual; colormap = :thermal, colorrange = field_limits)
        hm_error = heatmap!(ax_error, signed_error; colormap = :balance, colorrange = error_limits)

        stats = payload.summary
        ax_avg.title = @sprintf("%s\nmean %.3e", var, stats.mean_averaged)
        ax_manual.title = @sprintf("%s\nmean %.3e", var, stats.mean_manual)
        ax_error.title = @sprintf("%s\nRMS %.3e | max %.3e", var, stats.rms_error, stats.max_abs_error)

        Label(fig[plot_row, 4],
              @sprintf("N=%d\nfiles=%d\navg saves=%d", stats.sample_count, stats.source_file_count, stats.averaged_snapshot_count),
              fontsize = 14,
              tellwidth = false)
        Colorbar(fig[plot_row, 5], hm_error, label = "Error")
    end

    outpath = joinpath(ANALYSIS_OUTPUT_PATH, "manual_vs_averaged_surface_depth$(depth)_run$(run_label).$(FIGURE_FORMAT)")
    save(outpath, fig)
    return outpath
end

function main()
    if any(arg -> arg in ("-h", "--help"), ARGS)
        usage()
        return nothing
    end

    selected_run = isempty(ARGS) ? choose_run_id(OUTPUT_PATH) : parse(Int, ARGS[1])
    target_depth = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : TARGET_DEPTH

    averaged_by_depth = files_by_depth(averaged_depth_files(OUTPUT_PATH), selected_run)
    instantaneous_by_depth = files_by_depth(instantaneous_depth_files(OUTPUT_PATH), selected_run)
    common_depths = sort!(collect(intersect(keys(averaged_by_depth), keys(instantaneous_by_depth))))

    isempty(common_depths) && error("No common depth files found for run $(lpad(string(selected_run), 4, '0'))")
    target_depth in common_depths || error("Depth level $(target_depth) not found for run $(lpad(string(selected_run), 4, '0')); available depths: $(common_depths)")

    averaged_file = averaged_by_depth[target_depth]
    instantaneous_file = instantaneous_by_depth[target_depth]

    averaged_vars = Set(timeseries_variable_names(averaged_file))
    instantaneous_vars = Set(timeseries_variable_names(instantaneous_file))
    compare_vars = sort!(collect(intersect(averaged_vars, instantaneous_vars)))
    isempty(compare_vars) && error("No common variables found between averaged and instantaneous files.")

    @info "Comparing manual instantaneous averages to AveragedTimeInterval outputs" run_id = lpad(string(selected_run), 4, '0') depth = target_depth variables = compare_vars

    summary_rows = NamedTuple[]
    result_payload = Dict{String, Any}()

    for var in compare_vars
        summary, payload = compare_depth_variable(instantaneous_file, averaged_file, var)
        row = merge((; depth = target_depth, variable = var), summary)
        push!(summary_rows, row)
        result_payload[var] = merge(payload, (; summary = row))
    end

    run_label = lpad(string(selected_run), 4, '0')
    csv_path = joinpath(ANALYSIS_OUTPUT_PATH, "manual_vs_averaged_summary_depth$(target_depth)_run$(run_label).csv")
    jld2_path = joinpath(ANALYSIS_OUTPUT_PATH, "manual_vs_averaged_fields_depth$(target_depth)_run$(run_label).jld2")
    figure_path = save_surface_comparison_figure(compare_vars, result_payload, target_depth, run_label)

    write_summary_csv(csv_path, summary_rows)
    jldsave(jld2_path; run_id = selected_run, depth = target_depth, variables = compare_vars, results = result_payload)

    print_summary(summary_rows)
    @info "Saved comparison outputs." csv_path jld2_path figure_path

    return nothing
end

main()
