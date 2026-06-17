using CairoMakie
using NumericalEarth
using ConservativeRegridding
using Dates
using Glob
using JLD2
using Oceananigans
using Statistics
using WorldOceanAtlasTools

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

with_trailing_slash(path::AbstractString) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator

const OUTPUT_PATH = with_trailing_slash(expanduser(get(ENV, "OUTPUT_PATH", "/home/tsohail/uom/ocean-ensembles/outputs/")))
const FIGDIR = with_trailing_slash(expanduser(get(ENV, "FIGDIR", "/home/tsohail/uom/ocean-ensembles/figures/")))
const ANALYSIS_OUTPUT_PATH = with_trailing_slash(expanduser(get(ENV, "COMPARE_2D_OUTPUT_PATH", OUTPUT_PATH)))
const RESOLUTION = get(ENV, "COMPARE_2D_RESOLUTION", "sxtdeg")
const VIDEO_FRAMERATE = parse(Int, get(ENV, "COMPARE_2D_FRAMERATE", "3"))
const VIDEO_SIM_YEARS_PER_SECOND = parse(Float64, get(ENV, "COMPARE_2D_SIM_YEARS_PER_SECOND", "0.2"))
const MIN_VIDEO_FRAMERATE = 1.0
const MAX_VIDEO_FRAMERATE = 60.0
const NAN_PLOT_COLOR = :lightgray

const NOLEAP_MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
const SECONDS_PER_DAY = 86400.0
const SECONDS_PER_YEAR = sum(NOLEAP_MONTH_DAYS) * SECONDS_PER_DAY
const CUMULATIVE_MONTH_SECONDS = cumsum(vcat(0, NOLEAP_MONTH_DAYS)) .* SECONDS_PER_DAY
const RYF_YEAR_DAYS = 365.0
const PROGRESS_UPDATES = 20
const GC_INTERVAL = 12
const VIDEO_CHUNK_SIZE = 120
const DEPTH_FILE_RUN_PREFIX = "_fields_$(RESOLUTION)_RYF_run"
const DEFAULT_COMPARISON_VARS = ["T", "S"]

run_number(path::AbstractString) = begin
    m = match(r"_run(\d+)\.jld2$", basename(path))
    isnothing(m) ? -1 : parse(Int, m.captures[1])
end

depth_level(path::AbstractString) = begin
    m = match(r"global_(\d+)_fields_", basename(path))
    isnothing(m) ? -1 : parse(Int, m.captures[1])
end

numeric_timeseries_keys(group) = sort(parse.(Int, filter(k -> tryparse(Int, k) !== nothing, collect(keys(group)))))

@inline function extract_2d(raw)
    if ndims(raw) == 2
        return raw
    elseif ndims(raw) == 3
        return view(raw, :, :, 1)
    end

    return nothing
end

@inline function extract_2d_f32(raw)
    src = extract_2d(raw)
    src === nothing && return nothing
    return Float32.(src)
end

function copy_2d_to!(dest::Matrix{Float32}, raw)
    src = extract_2d(raw)
    src === nothing && return false
    size(dest) == size(src) || return false

    @inbounds for i in eachindex(dest, src)
        dest[i] = Float32(src[i])
    end
    return true
end

function month_of_year_and_start(t::Real)
    year_index = floor(Int, t / SECONDS_PER_YEAR)
    second_of_year = t - year_index * SECONDS_PER_YEAR
    month_of_year = searchsortedlast(CUMULATIVE_MONTH_SECONDS, second_of_year + eps(second_of_year))
    month_of_year = clamp(month_of_year, 1, 12)
    month_start = year_index * SECONDS_PER_YEAR + CUMULATIVE_MONTH_SECONDS[month_of_year]
    month_stop = year_index * SECONDS_PER_YEAR + CUMULATIVE_MONTH_SECONDS[month_of_year + 1]
    return year_index, month_of_year, month_start, month_stop
end

function split_interval_by_month(t0::Real, t1::Real)
    t1 <= t0 && return Tuple{Int, Int, Float64, Float64}[]

    pieces = Tuple{Int, Int, Float64, Float64}[]
    current = Float64(t0)
    stop = Float64(t1)

    while current < stop
        year_index, month_of_year, _, month_stop = month_of_year_and_start(current)
        piece_stop = min(stop, month_stop)
        push!(pieces, (year_index, month_of_year, current, piece_stop))
        current = piece_stop
    end

    return pieces
end

ryf_month_starts_days() = Float64.(cumsum((0, NOLEAP_MONTH_DAYS[1:end-1]...)))

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

function load_grid_from_output_file(filepath::AbstractString)
    return jldopen(filepath, "r") do f
        read_serialized_grid(f)
    end
end

function depth_slice_files(path::AbstractString, resolution::AbstractString)
    files = glob("combined_*_fields_$(resolution)_RYF_run*.jld2", path)
    files = filter(files) do f
        !occursin("_rank", f) &&
        occursin(r"global_\d+_fields_", basename(f)) &&
        run_number(f) >= 0
    end
    sort!(files; by = f -> (depth_level(f), run_number(f)))
    return files
end

function log_progress(tag::String, current::Int, total::Int)
    width = 24
    fraction = current / total
    filled = clamp(floor(Int, width * fraction), 0, width)
    bar = "[" * repeat("=", filled) * repeat(".", width - filled) * "]"
    percent = round(100 * fraction; digits = 1)
    @info "Progress." tag current total percent bar
    return nothing
end

function index_depth_tracer_series(depth::Int; path::AbstractString = OUTPUT_PATH, resolution::AbstractString = RESOLUTION)
    files = filter(f -> depth_level(f) == depth, depth_slice_files(path, resolution))
    isempty(files) && error("No combined depth files found for depth index $(depth).")

    records_by_time = Dict{Float64, NamedTuple{(:run, :filepath, :key), Tuple{Int, String, Int}}}()
    replaced_duplicates = 0

    for (file_index, file) in enumerate(files)
        run = run_number(file)

        jldopen(file, "r") do f
            haskey(f, "timeseries/t") || return
            haskey(f, "timeseries/T") || return
            haskey(f, "timeseries/S") || return

            ts_keys = numeric_timeseries_keys(f["timeseries/t"])
            for key in ts_keys
                tval = Float64(f["timeseries/t/$key"])
                existing = get(records_by_time, tval, nothing)
                if isnothing(existing) || run >= existing.run
                    replaced_duplicates += !isnothing(existing) && run > existing.run ? 1 : 0
                    records_by_time[tval] = (run = run, filepath = file, key = key)
                end
            end
        end

        if file_index == 1 || file_index == length(files) || file_index % max(1, cld(length(files), PROGRESS_UPDATES)) == 0
            log_progress("index_depth_$(depth)", file_index, length(files))
        end
    end

    times = sort(collect(keys(records_by_time)))
    isempty(times) && error("No tracer timesteps found after deduplication for depth $(depth).")
    refs = [(time = t, run = records_by_time[t].run, filepath = records_by_time[t].filepath, key = records_by_time[t].key) for t in times]
    grid = load_grid_from_output_file(first(files))

    @info "Indexed depth tracer timeseries." depth frames = length(times) replaced_duplicates
    return (; times, refs, grid)
end

function load_depth_tracer_frame(ref, var::String)
    return jldopen(ref.filepath, "r") do f
        haskey(f, "timeseries/$var") || return nothing
        extract_2d_f32(f["timeseries/$var/$(ref.key)"])
    end
end

function monthly_average_field_refs(times::AbstractVector{<:Real}, refs, var::String)
    length(times) == length(refs) || error("Times and field references must have the same length.")
    length(times) >= 2 || error("Need at least two samples for dt-weighted monthly averages.")

    weighted_sum = Dict{Int, Matrix{Float64}}()
    total_dt = Dict{Int, Float64}()

    for n in 1:length(times)-1
        field = load_depth_tracer_frame(refs[n], var)
        field === nothing && error("Could not load variable=$(var) from $(refs[n].filepath) key $(refs[n].key).")
        field64 = Float64.(field)

        for (year_index, month_of_year, piece_start, piece_stop) in split_interval_by_month(times[n], times[n+1])
            bin = 12 * year_index + month_of_year
            dt = piece_stop - piece_start
            if !haskey(weighted_sum, bin)
                weighted_sum[bin] = zeros(Float64, size(field64))
            end
            weighted_sum[bin] .+= field64 .* dt
            total_dt[bin] = get(total_dt, bin, 0.0) + dt
        end
    end

    bins = sort(collect(keys(weighted_sum)))
    years = [fld(bin - 1, 12) for bin in bins]
    months = [mod1(bin, 12) for bin in bins]

    return (; bins, years, months, total_dt, weighted_sum)
end

function write_monthly_field_cache!(filepath::AbstractString, monthly_T, monthly_S)
    jldopen(filepath, "w") do f
        f["bins"] = monthly_T.bins
        f["years"] = monthly_T.years
        f["months"] = monthly_T.months
        f["total_dt"] = [monthly_T.total_dt[bin] for bin in monthly_T.bins]

        for (frame, bin) in enumerate(monthly_T.bins)
            f["T/$frame"] = Float32.(monthly_T.weighted_sum[bin] ./ monthly_T.total_dt[bin])
            f["S/$frame"] = Float32.(monthly_S.weighted_sum[bin] ./ monthly_S.total_dt[bin])
        end
    end

    return nothing
end

function load_cached_frame(filepath::AbstractString, var::String, index::Int)
    return jldopen(filepath, "r") do f
        extract_2d_f32(f["$var/$index"])
    end
end

function cached_frame_reader(filepath::AbstractString)
    handle = Ref{Any}(nothing)

    function ensure_handle()
        if handle[] === nothing
            handle[] = jldopen(filepath, "r")
        end
        return handle[]
    end

    function load_frame!(dest::Matrix{Float32}, var::String, index::Int)
        file = ensure_handle()
        return copy_2d_to!(dest, file["$var/$index"])
    end

    function close_reader!()
        handle[] !== nothing && close(handle[])
        handle[] = nothing
        return nothing
    end

    return (; load_frame!, close_reader!)
end

function load_woa_monthly_fields()
    arch = CPU()
    T_fields = Vector{Any}(undef, 12)
    S_fields = Vector{Any}(undef, 12)

    for month in 1:12
        date = DateTime(2018, month, 1)
        T_fields[month] = Field(Metadatum(:temperature; date, dataset = WOAMonthly()), arch; inpainting = nothing, cache_inpainted_data = false)
        S_fields[month] = Field(Metadatum(:salinity; date, dataset = WOAMonthly()), arch; inpainting = nothing, cache_inpainted_data = false)
    end

    return T_fields, S_fields
end

function wet_clean_fields(T, S)
    T_raw = Float64.(Array(interior(T)))
    S_raw = Float64.(Array(interior(S)))
    wet = isfinite.(T_raw) .& isfinite.(S_raw)
    T_clean = ifelse.(wet, T_raw, 0.0)
    S_clean = ifelse.(wet, S_raw, 0.0)
    return T_clean, S_clean, Float64.(wet)
end

function overlap_coefficients(source_faces::AbstractVector{<:Real}, target_faces::AbstractVector{<:Real})
    source_faces = Float64.(source_faces)
    target_faces = Float64.(target_faces)

    nt = length(target_faces) - 1
    coeffs = Vector{Tuple{Vector{Int}, Vector{Float64}}}(undef, nt)

    for j in 1:nt
        a = target_faces[j]
        b = target_faces[j + 1]
        Δ = b - a
        Δ > 0 || error("Target faces must be strictly increasing.")

        idxs = Int[]
        weights = Float64[]

        for i in 1:length(source_faces)-1
            left = max(a, source_faces[i])
            right = min(b, source_faces[i + 1])
            overlap = right - left
            if overlap > 0
                push!(idxs, i)
                push!(weights, overlap / Δ)
            end
        end

        coeffs[j] = (idxs, weights)
    end

    return coeffs
end

function vertically_remap_month_to_depth(T_field, S_field, coeff::Tuple{Vector{Int}, Vector{Float64}})
    idxs, weights = coeff
    T_clean, S_clean, wet = wet_clean_fields(T_field, S_field)

    nx, ny, _ = size(T_clean)
    T_num = zeros(Float64, nx, ny)
    S_num = zeros(Float64, nx, ny)
    wet_sum = zeros(Float64, nx, ny)

    for (index, weight) in zip(idxs, weights)
        @views begin
            wet_k = wet[:, :, index]
            wet_sum .+= weight .* wet_k
            T_num .+= weight .* T_clean[:, :, index] .* wet_k
            S_num .+= weight .* S_clean[:, :, index] .* wet_k
        end
    end

    T_target = Matrix{Float32}(undef, nx, ny)
    S_target = Matrix{Float32}(undef, nx, ny)

    @inbounds for i in eachindex(wet_sum, T_target, S_target, T_num, S_num)
        if wet_sum[i] > 0
            T_target[i] = Float32(T_num[i] / wet_sum[i])
            S_target[i] = Float32(S_num[i] / wet_sum[i])
        else
            T_target[i] = NaN32
            S_target[i] = NaN32
        end
    end

    return T_target, S_target
end

function centered_2d_field(grid)
    return Field{Center, Center, Nothing}(grid)
end

function set_2d_field!(field, A::AbstractMatrix)
    set!(field, reshape(Float64.(A), size(A, 1), size(A, 2), 1))
    return field
end

function conservative_horizontal_regrid(A::AbstractMatrix,
                                        wet::AbstractMatrix,
                                        value_src,
                                        wet_src,
                                        value_dst,
                                        wet_dst,
                                        regridder)
    fill_value = ifelse.(isfinite.(A), Float32.(A), 0f0)
    fill_wet = Float32.(wet)

    set_2d_field!(value_src, fill_value)
    set_2d_field!(wet_src, fill_wet)
    ConservativeRegridding.regrid!(value_dst, regridder, value_src)
    ConservativeRegridding.regrid!(wet_dst, regridder, wet_src)

    value_arr = Float32.(Array(interior(value_dst)))
    wet_arr = Float32.(Array(interior(wet_dst)))
    ndims(value_arr) == 3 && (value_arr = value_arr[:, :, 1])
    ndims(wet_arr) == 3 && (wet_arr = wet_arr[:, :, 1])

    out = Matrix{Float32}(undef, size(value_arr))
    @inbounds for i in eachindex(out, value_arr, wet_arr)
        if wet_arr[i] > 0
            out[i] = value_arr[i] / wet_arr[i]
        else
            out[i] = NaN32
        end
    end

    return out
end

function bottom_height_matrix(grid)
    isnothing(grid) && return nothing
    hasproperty(grid, :immersed_boundary) || return nothing

    immersed_boundary = getproperty(grid, :immersed_boundary)
    hasproperty(immersed_boundary, :bottom_height) || return nothing

    bottom_height_field = getproperty(immersed_boundary, :bottom_height)
    return Float32.(Array(interior(bottom_height_field, :, :, 1)))
end

function depth_ocean_mask(z_center::Real, bottom_height::Union{Nothing, AbstractMatrix})
    isnothing(bottom_height) && return nothing
    return bottom_height .< Float32(z_center)
end

function apply_plot_mask(A::AbstractMatrix, plot_mask::Union{Nothing, AbstractMatrix{Bool}})
    masked = Float32.(A)
    isnothing(plot_mask) && return masked
    size(masked) == size(plot_mask) || error("Plot mask shape mismatch: got $(size(plot_mask)) expected $(size(masked)).")

    @inbounds for i in eachindex(masked, plot_mask)
        plot_mask[i] || (masked[i] = NaN32)
    end

    return masked
end

function apply_plot_mask!(dest::Matrix{Float32}, src::AbstractMatrix, plot_mask::Union{Nothing, AbstractMatrix{Bool}})
    size(dest) == size(src) || error("Destination/src size mismatch: $(size(dest)) vs $(size(src)).")
    if isnothing(plot_mask)
        copyto!(dest, src)
        return dest
    end

    size(dest) == size(plot_mask) || error("Plot mask shape mismatch: got $(size(plot_mask)) expected $(size(dest)).")
    @inbounds for i in eachindex(dest, src, plot_mask)
        dest[i] = plot_mask[i] ? Float32(src[i]) : NaN32
    end
    return dest
end

function sampled_reference_indices(n::Int; max_samples::Int = 24)
    n <= 0 && return Int[]
    n <= max_samples && return collect(1:n)
    return unique(round.(Int, range(1, n; length = max_samples)))
end

function error_colorrange(var::String,
                          frame_count::Int,
                          month_of_year::AbstractVector{<:Integer},
                          model_cache_files::Vector{String},
                          woa_cache_files::Vector{String},
                          depth_masks)
    maxabs = 0f0
    found = false
    sampled_frames = sampled_reference_indices(frame_count)

    @info "Sampling cached error frames to determine colorrange." variable = var sampled_frames = length(sampled_frames) depths = length(model_cache_files)

    for (sample_index, frame) in enumerate(sampled_frames)
        month = month_of_year[frame]
        for depth_index in eachindex(model_cache_files)
            model_frame = load_cached_frame(model_cache_files[depth_index], var, frame)
            woa_frame = load_cached_frame(woa_cache_files[depth_index], var, month)
            error_frame = build_error_frame(model_frame, woa_frame, depth_masks[depth_index])

            for value in error_frame
                if isfinite(value)
                    maxabs = max(maxabs, abs(Float32(value)))
                    found = true
                end
            end
        end

        if sample_index == 1 || sample_index == length(sampled_frames) || sample_index % max(1, cld(length(sampled_frames), PROGRESS_UPDATES)) == 0
            log_progress("colorrange_$(var)", sample_index, length(sampled_frames))
        end
    end

    if !found
        return var == "T" ? (-2f0, 2f0) : (-1f0, 1f0)
    elseif maxabs == 0
        return (-1f-6, 1f-6)
    end

    return (-maxabs, maxabs)
end

function build_monthly_model_caches(depth_levels::Vector{Int}; path::AbstractString = OUTPUT_PATH, resolution::AbstractString = RESOLUTION)
    cache_files = String[]
    bins = Int[]
    years = Int[]
    months = Int[]
    model_grid = nothing

    for (depth_index, depth) in enumerate(depth_levels)
        series = index_depth_tracer_series(depth; path, resolution)
        monthly_T = monthly_average_field_refs(series.times, series.refs, "T")
        monthly_S = monthly_average_field_refs(series.times, series.refs, "S")

        isempty(bins) || monthly_T.bins == bins || error("Temperature monthly bins do not align across depths.")
        isempty(bins) || monthly_S.bins == bins || error("Salinity monthly bins do not align across depths.")

        if isempty(bins)
            bins = monthly_T.bins
            years = monthly_T.years
            months = monthly_T.months
            model_grid = series.grid
        end

        cache_file = joinpath(ANALYSIS_OUTPUT_PATH, "compare_2d_WOA_model_depth$(depth)_$(resolution).jld2")
        write_monthly_field_cache!(cache_file, monthly_T, monthly_S)
        push!(cache_files, cache_file)

        if depth_index == 1 || depth_index == length(depth_levels) || depth_index % max(1, cld(length(depth_levels), PROGRESS_UPDATES)) == 0
            log_progress("monthly_model_cache", depth_index, length(depth_levels))
        end
    end

    return (; cache_files, bins, years, months, model_grid)
end

function build_woa_depth_cache(model_grid, depth_levels::Vector{Int}; resolution::AbstractString = RESOLUTION)
    T_fields, S_fields = load_woa_monthly_fields()
    woa_grid = T_fields[1].grid
    model_underlying = hasproperty(model_grid, :underlying_grid) ? model_grid.underlying_grid : model_grid

    source_z_faces = Float64.(collect(woa_grid.z.cᵃᵃᶠ))
    model_z_faces = Float64.(collect(model_underlying.z.cᵃᵃᶠ))
    model_z_centers = Float64.(collect(model_underlying.z.cᵃᵃᶜ))

    target_faces = [model_z_faces[depth:depth+1] for depth in depth_levels]
    target_coeffs = [only(overlap_coefficients(source_z_faces, faces)) for faces in target_faces]

    # ConservativeRegridding in this repo already supports native-grid regridding
    # between LatitudeLongitudeGrid and TripolarGrid fields. Keep that path and let
    # set_2d_field! promote the matrix data to Float64, rather than reconstructing
    # synthetic lat-lon grids from center coordinates.
    src_value = centered_2d_field(woa_grid)
    src_wet = centered_2d_field(woa_grid)
    dst_value = centered_2d_field(model_underlying)
    dst_wet = centered_2d_field(model_underlying)

    manifold = ConservativeRegridding.Spherical(6.371e6)
    regridder = ConservativeRegridding.Regridder(manifold, dst_value, src_value)

    cache_files = String[]

    for (depth_index, coeff) in enumerate(target_coeffs)
        cache_file = joinpath(ANALYSIS_OUTPUT_PATH, "compare_2d_WOA_woa_depth$(depth_levels[depth_index])_$(resolution).jld2")
        jldopen(cache_file, "w") do f
            for month in 1:12
                T_vertical, S_vertical = vertically_remap_month_to_depth(T_fields[month], S_fields[month], coeff)
                wet = isfinite.(T_vertical) .& isfinite.(S_vertical)
                f["T/$month"] = conservative_horizontal_regrid(T_vertical, wet, src_value, src_wet, dst_value, dst_wet, regridder)
                f["S/$month"] = conservative_horizontal_regrid(S_vertical, wet, src_value, src_wet, dst_value, dst_wet, regridder)
            end
        end
        push!(cache_files, cache_file)

        if depth_index == 1 || depth_index == length(depth_levels) || depth_index % max(1, cld(length(depth_levels), PROGRESS_UPDATES)) == 0
            log_progress("woa_depth_cache", depth_index, length(depth_levels))
        end
    end

    return (; cache_files, model_z_centers, model_z_faces)
end

function month_time_metadata(years::AbstractVector{<:Integer}, months::AbstractVector{<:Integer})
    time_days = years .* RYF_YEAR_DAYS .+ ryf_month_starts_days()[months] .+ 0.5 .* NOLEAP_MONTH_DAYS[months]
    return (; month_of_year = months, year_index = years, time_days)
end

function constant_model_dt_framerate(times::Vector{Float64};
                                     fallback::Real = VIDEO_FRAMERATE,
                                     sim_years_per_second::Real = VIDEO_SIM_YEARS_PER_SECOND)
    length(times) > 1 || return Int(round(fallback))
    Δts = [times[i + 1] - times[i] for i in 1:length(times)-1 if times[i + 1] > times[i]]
    isempty(Δts) && return Int(round(fallback))

    fps = sim_years_per_second * SECONDS_PER_YEAR / median(Δts)
    return round(Int, clamp(fps, MIN_VIDEO_FRAMERATE, MAX_VIDEO_FRAMERATE))
end

function panel_layout(npanels::Int)
    npanels > 0 || error("At least one panel is required.")
    if npanels <= 3
        return 1, npanels
    elseif npanels == 4
        return 2, 2
    else
        return cld(npanels, 3), 3
    end
end

sanitize_var_token(var::String) = replace(var, r"[^A-Za-z0-9]+" => "-")
vars_slug(vars::Vector{String}) = join(sanitize_var_token.(vars), "_")

function resolve_ffmpeg_binary()
    ffmpeg = Sys.which("ffmpeg")
    !isnothing(ffmpeg) && return ffmpeg

    fallback = "/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/13.3.0/FFmpeg/7.0.2/bin/ffmpeg"
    return isfile(fallback) ? fallback : nothing
end

function record_video_in_chunks(render_frame!,
                                fig,
                                outname::AbstractString,
                                frames,
                                framerate::Int;
                                chunk_size::Int = VIDEO_CHUNK_SIZE,
                                tag::AbstractString = "video",
                                chunk_cleanup::Union{Nothing, Function} = nothing)
    isempty(frames) && error("No frames were provided for chunked recording.")
    mkpath(dirname(outname))

    ffmpeg = resolve_ffmpeg_binary()
    isnothing(ffmpeg) && error("Could not find ffmpeg for chunked video assembly.")

    chunk_dir = mktempdir(dirname(outname); prefix = basename(outname) * "_chunks_")
    chunk_paths = String[]

    try
        for (chunk_index, start_idx) in enumerate(1:chunk_size:length(frames))
            stop_idx = min(length(frames), start_idx + chunk_size - 1)
            chunk_frames = collect(frames[start_idx:stop_idx])
            chunk_path = joinpath(chunk_dir, "chunk_" * lpad(string(chunk_index), 4, '0') * ".mp4")
            push!(chunk_paths, chunk_path)
            @info "Recording video chunk." tag chunk_index start_idx stop_idx chunk_path
            record(fig, chunk_path, chunk_frames; framerate = framerate) do frame
                render_frame!(frame)
            end
            # Cleanup between chunks
            if !isnothing(chunk_cleanup)
                chunk_cleanup()
            end
            # Aggressively clear memory between chunks
            chunk_frames = nothing
            GC.gc(false)
            GC.gc()
            GC.gc()
        end

        concat_file = joinpath(chunk_dir, "concat.txt")
        open(concat_file, "w") do io
            for chunk_path in chunk_paths
                println(io, "file '" * replace(abspath(chunk_path), "'" => "'\''") * "'")
            end
        end

        cmd = `$ffmpeg -y -f concat -safe 0 -i $concat_file -c copy $outname`
        @info "Concatenating video chunks." tag outname chunk_count = length(chunk_paths) ffmpeg
        run(cmd)
    finally
        rm(chunk_dir; recursive = true, force = true)
    end

    return outname
end

function validate_requested_variables(requested_vars::Vector{String}, available_vars::Vector{String})
    isempty(requested_vars) && throw(ArgumentError("At least one variable must be requested."))
    missing = [var for var in requested_vars if !(var in available_vars)]
    isempty(missing) || error("Requested variables not available: $(missing). Available variables: $(available_vars)")
    return requested_vars
end

function single_depth_level(k::Union{Nothing, Int}, available_depths::Vector{Int})
    isempty(available_depths) && error("No depth levels are available.")
    isnothing(k) && return first(available_depths)
    k in available_depths || error("Requested depth level k=$(k) not available. Available levels: $(available_depths)")
    return k
end

function build_error_frame(model_frame::AbstractMatrix,
                           woa_frame::AbstractMatrix,
                           plot_mask::Union{Nothing, AbstractMatrix{Bool}})
    out = Matrix{Float32}(undef, size(model_frame))

    @inbounds for i in eachindex(out, model_frame, woa_frame)
        if isfinite(model_frame[i]) && isfinite(woa_frame[i])
            out[i] = Float32(model_frame[i] - woa_frame[i])
        else
            out[i] = NaN32
        end
    end

    return apply_plot_mask(out, plot_mask)
end

function build_error_frame!(dest::Matrix{Float32},
                            model_frame::AbstractMatrix,
                            woa_frame::AbstractMatrix,
                            plot_mask::Union{Nothing, AbstractMatrix{Bool}})
    size(dest) == size(model_frame) == size(woa_frame) || error("Error-frame size mismatch.")
    if !isnothing(plot_mask)
        size(dest) == size(plot_mask) || error("Plot mask shape mismatch: got $(size(plot_mask)) expected $(size(dest)).")
    end

    @inbounds for i in eachindex(dest, model_frame, woa_frame)
        if isfinite(model_frame[i]) && isfinite(woa_frame[i]) && (isnothing(plot_mask) || plot_mask[i])
            dest[i] = Float32(model_frame[i] - woa_frame[i])
        else
            dest[i] = NaN32
        end
    end

    return dest
end

function error_colorrange_for_depth(var::String,
                                    frame_count::Int,
                                    month_of_year::AbstractVector{<:Integer},
                                    model_cache_file::AbstractString,
                                    woa_cache_file::AbstractString,
                                    depth_mask)
    maxabs = 0f0
    found = false
    sampled_frames = sampled_reference_indices(frame_count)

    @info "Sampling cached error frames to determine colorrange." variable = var sampled_frames = length(sampled_frames)

    for (sample_index, frame) in enumerate(sampled_frames)
        month = month_of_year[frame]
        model_frame = load_cached_frame(model_cache_file, var, frame)
        woa_frame = load_cached_frame(woa_cache_file, var, month)
        error_frame = build_error_frame(model_frame, woa_frame, depth_mask)

        for value in error_frame
            if isfinite(value)
                maxabs = max(maxabs, abs(Float32(value)))
                found = true
            end
        end

        if sample_index == 1 || sample_index == length(sampled_frames) || sample_index % max(1, cld(length(sampled_frames), PROGRESS_UPDATES)) == 0
            log_progress("colorrange_$(var)", sample_index, length(sampled_frames))
        end
    end

    if !found
        return var == "T" ? (-2f0, 2f0) : (-1f0, 1f0)
    elseif maxabs == 0
        return (-1f-6, 1f-6)
    end

    return (-maxabs, maxabs)
end

function make_error_multivariable_video(vars::Vector{String},
                                        metadata,
                                        depth::Int,
                                        depth_mask,
                                        model_z_centers::AbstractVector{<:Real},
                                        colorranges::Dict{String, Tuple{Float32, Float32}},
                                        model_cache_file::AbstractString,
                                        woa_cache_file::AbstractString;
                                        outname::AbstractString,
                                        framerate::Union{Nothing, Real} = nothing)
    nt = length(metadata.month_of_year)
    nrows, ncols = panel_layout(length(vars))
    depth_value = round(abs(Float64(model_z_centers[depth])); digits = 1)

    initial_fields = Dict{String, Matrix{Float32}}()
    month1 = metadata.month_of_year[1]
    for var in vars
        model_frame = load_cached_frame(model_cache_file, var, 1)
        woa_frame = load_cached_frame(woa_cache_file, var, month1)
        initial_fields[var] = build_error_frame(model_frame, woa_frame, depth_mask)
    end
    model_buffers = Dict(var => similar(initial_fields[var]) for var in vars)
    woa_buffers = Dict(var => similar(initial_fields[var]) for var in vars)
    model_reader = cached_frame_reader(model_cache_file)
    woa_reader = cached_frame_reader(woa_cache_file)

    fig = Figure(size = (520 * ncols, 400 * nrows))
    title = Label(fig[0, :], "", tellwidth = false)
    observables = Dict(var => Observable(initial_fields[var]) for var in vars)

    for (panel_index, var) in enumerate(vars)
        row = cld(panel_index, ncols)
        col = mod1(panel_index, ncols)
        layout_row = 2 * row - 1
        panel_title = var == "T" ? "Temperature error vs WOA (degC)" : "Salinity error vs WOA"
        ax = Axis(fig[layout_row, col], title = panel_title)
        hm = heatmap!(ax, observables[var], colormap = :balance, colorrange = colorranges[var], nan_color = NAN_PLOT_COLOR)
        Colorbar(fig[layout_row + 1, col], hm, vertical = false)
    end

    resize_to_layout!(fig)
    progress_step = max(1, cld(nt, PROGRESS_UPDATES))
    times = Float64.(metadata.time_days) .* SECONDS_PER_DAY
    framerate = isnothing(framerate) ? constant_model_dt_framerate(times) : framerate

    try
        record_video_in_chunks(fig, outname, 1:nt, framerate; 
                              tag = "compare_2d_WOA_k$(depth)",
                              chunk_cleanup = () -> begin
                                  # Close readers between chunks to release file handles
                                  model_reader.close_reader!()
                                  woa_reader.close_reader!()
                                  # Recreate readers for next chunk
                                  model_reader = cached_frame_reader(model_cache_file)
                                  woa_reader = cached_frame_reader(woa_cache_file)
                                  # Clear buffers
                                  for var in vars
                                      fill!(model_buffers[var], NaN32)
                                      fill!(woa_buffers[var], NaN32)
                                  end
                              end) do frame
            year_index = metadata.year_index[frame]
            month = metadata.month_of_year[frame]
            title.text = "WOA comparison error | Depth k=$(depth) ($(depth_value) m) | RYF year $(year_index + 1) | $(Dates.format(Date(2001, month, 1), "mmm"))"

            for var in vars
                model_reader.load_frame!(model_buffers[var], var, frame) || error("Could not load model cache frame $(frame) for variable=$(var).")
                woa_reader.load_frame!(woa_buffers[var], var, month) || error("Could not load WOA cache frame $(month) for variable=$(var).")
                build_error_frame!(observables[var][], model_buffers[var], woa_buffers[var], depth_mask)
                notify(observables[var])
            end

            if frame % GC_INTERVAL == 0
                GC.gc(false)
            end
            if frame % (5 * GC_INTERVAL) == 0
                GC.gc()
            end
            if frame == 1 || frame == nt || frame % progress_step == 0
                log_progress("video_compare_2d_WOA", frame, nt)
            end
        end
    finally
        model_reader.close_reader!()
        woa_reader.close_reader!()
    end

    @info "Saved WOA error animation." variables = vars outname frames = nt framerate
    return outname
end

function main()
    return main(DEFAULT_COMPARISON_VARS)
end

function main(vars::Vector{String}; k::Union{Nothing, Int} = nothing)
    mkpath(FIGDIR)
    mkpath(ANALYSIS_OUTPUT_PATH)

    files = depth_slice_files(OUTPUT_PATH, RESOLUTION)
    isempty(files) && error("No combined depth files found for resolution=$(RESOLUTION) in $(OUTPUT_PATH).")

    depth_levels = sort(unique(filter(>=(0), depth_level.(files))))
    isempty(depth_levels) && error("Could not infer any depth levels from combined depth files.")
    validate_requested_variables(vars, DEFAULT_COMPARISON_VARS)
    requested_depth = single_depth_level(k, depth_levels)

    @info "Building monthly model caches for depth slices." depth_count = length(depth_levels) output_path = OUTPUT_PATH
    model_cache = build_monthly_model_caches(depth_levels; path = OUTPUT_PATH, resolution = RESOLUTION)
    metadata = month_time_metadata(model_cache.years, model_cache.months)
    model_grid = model_cache.model_grid
    isnothing(model_grid) && error("No serialized grid found in the combined depth files.")

    @info "Building WOA depth caches on the model grid." depth_count = length(depth_levels)
    woa_cache = build_woa_depth_cache(model_grid, depth_levels; resolution = RESOLUTION)

    depth_index = findfirst(==(requested_depth), depth_levels)
    isnothing(depth_index) && error("Requested depth k=$(requested_depth) was not found in depth_levels.")

    bottom_height = bottom_height_matrix(model_grid)
    depth_mask = depth_ocean_mask(woa_cache.model_z_centers[requested_depth], bottom_height)

    colorranges = Dict{String, Tuple{Float32, Float32}}()
    for var in vars
        colorranges[var] = error_colorrange_for_depth(var, length(model_cache.bins), metadata.month_of_year,
                                                      model_cache.cache_files[depth_index], woa_cache.cache_files[depth_index], depth_mask)
    end

    video_file = joinpath(FIGDIR, "compare_2d_WOA_k$(requested_depth)_$(vars_slug(vars))_$(RESOLUTION).mp4")
    summary_file = joinpath(ANALYSIS_OUTPUT_PATH, "compare_2d_WOA_k$(requested_depth)_$(vars_slug(vars))_$(RESOLUTION).jld2")

    make_error_multivariable_video(vars, metadata, requested_depth, depth_mask,
                                   woa_cache.model_z_centers, colorranges,
                                   model_cache.cache_files[depth_index], woa_cache.cache_files[depth_index];
                                   outname = video_file)

    @info "Saving 2D WOA comparison metadata." summary_file
    jldsave(summary_file;
            resolution = RESOLUTION,
            depth_levels,
            requested_depth,
            variables = vars,
            model_z_centers = woa_cache.model_z_centers,
            model_z_faces = woa_cache.model_z_faces,
            month_bin = model_cache.bins,
            month_of_year = metadata.month_of_year,
            year_index = metadata.year_index,
            time_days = metadata.time_days,
            colorranges,
            video_file,
            model_cache_file = model_cache.cache_files[depth_index],
            woa_cache_file = woa_cache.cache_files[depth_index])

    @info "Finished building 2D WOA comparison animation." video_file summary_file nmonths = length(model_cache.bins) depth = requested_depth variables = vars
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
