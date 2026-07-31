# Shared, side-effect-free helpers for 2D comparison analysis scripts.

using CairoMakie
using Dates
using Glob
using JLD2
using Logging
using Oceananigans
using Statistics: median

with_trailing_slash(path::AbstractString) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator

const OUTPUT_PATH = with_trailing_slash(expanduser(get(ENV, "OUTPUT_PATH", "/home/tsohail/uom/ocean-ensembles/outputs/saved/")))
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
const PROGRESS_UPDATES = 20
const GC_INTERVAL = 12
const VIDEO_CHUNK_SIZE = 120
const ERROR_COLORRANGE_STD_MULTIPLIER = parse(Float32, get(ENV, "COMPARE_2D_ERROR_COLORRANGE_STD_MULTIPLIER", "1.0"))
const VALUE_COLORRANGE_STD_MULTIPLIER = parse(Float32, get(ENV, "COMPARE_2D_VALUE_COLORRANGE_STD_MULTIPLIER", "1.0"))

run_number(path::AbstractString) = begin
    match_result = match(r"_run(\d+)\.jld2$", basename(path))
    isnothing(match_result) ? -1 : parse(Int, match_result.captures[1])
end

numeric_timeseries_keys(group) = sort(parse.(Int, filter(key -> tryparse(Int, key) !== nothing, String.(collect(keys(group))))))

@inline function extract_2d(raw)
    ndims(raw) == 2 && return raw
    ndims(raw) == 3 && return view(raw, :, :, 1)
    return nothing
end

@inline function extract_2d_f32(raw)
    source = extract_2d(raw)
    source === nothing && return nothing
    return Float32.(source)
end

function copy_2d_to!(destination::Matrix{Float32}, raw)
    source = extract_2d(raw)
    source === nothing && return false
    size(destination) == size(source) || return false
    copyto!(destination, Float32.(source))
    return true
end

const OFFSET_ARRAYS = Base.loaded_modules[Base.PkgId(Base.UUID("6fe1bfb0-de20-5000-8ca7-80f57d26f881"), "OffsetArrays")]
underlying_grid(grid) = hasproperty(grid, :underlying_grid) ? getproperty(grid, :underlying_grid) : grid

function materialize_offset_array(source)
    if hasproperty(source, :parent) && hasproperty(source, :offsets)
        return OFFSET_ARRAYS.OffsetArray(getproperty(source, :parent), getproperty(source, :offsets)...)
    end
    return source
end

parent_array(source) = hasproperty(source, :parent) ? getproperty(source, :parent) : Array(source)

function interior_start(source, dim, fallback_halo, interior_size, stored_size)
    if hasproperty(source, :offsets)
        offsets = getproperty(source, :offsets)
        if dim <= length(offsets)
            start = 1 - offsets[dim]
            1 <= start <= stored_size - interior_size + 1 && return start
        end
    end
    stored_size == interior_size && return 1
    start = fallback_halo + 1
    1 <= start <= stored_size - interior_size + 1 && return start
    error("Could not crop stored grid data to its interior.")
end

function physical_matrix(source, grid; T = Float64)
    Nx, Ny = grid.Nx, grid.Ny
    data = parent_array(source)
    i0 = interior_start(source, 1, grid.Hx, Nx, size(data, 1))
    j0 = interior_start(source, 2, grid.Hy, Ny, size(data, 2))
    ndims(data) == 2 && return T.(view(data, i0:i0+Nx-1, j0:j0+Ny-1))
    ndims(data) == 3 && return T.(view(data, i0:i0+Nx-1, j0:j0+Ny-1, 1))
    error("Expected 2D or singleton-vertical grid data.")
end

function materialized_underlying_grid(grid)
    grid = underlying_grid(grid)
    return OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(CPU(),
        grid.Nx, grid.Ny, grid.Nz, grid.Hx, grid.Hy, grid.Hz, grid.Lz,
        materialize_offset_array(grid.λᶜᶜᵃ), materialize_offset_array(grid.λᶠᶜᵃ),
        materialize_offset_array(grid.λᶜᶠᵃ), materialize_offset_array(grid.λᶠᶠᵃ),
        materialize_offset_array(grid.φᶜᶜᵃ), materialize_offset_array(grid.φᶠᶜᵃ),
        materialize_offset_array(grid.φᶜᶠᵃ), materialize_offset_array(grid.φᶠᶠᵃ),
        grid.z,
        materialize_offset_array(grid.Δxᶜᶜᵃ), materialize_offset_array(grid.Δxᶠᶜᵃ),
        materialize_offset_array(grid.Δxᶜᶠᵃ), materialize_offset_array(grid.Δxᶠᶠᵃ),
        materialize_offset_array(grid.Δyᶜᶜᵃ), materialize_offset_array(grid.Δyᶠᶜᵃ),
        materialize_offset_array(grid.Δyᶜᶠᵃ), materialize_offset_array(grid.Δyᶠᶠᵃ),
        materialize_offset_array(grid.Azᶜᶜᵃ), materialize_offset_array(grid.Azᶠᶜᵃ),
        materialize_offset_array(grid.Azᶜᶠᵃ), materialize_offset_array(grid.Azᶠᶠᵃ),
        grid.radius, grid.conformal_mapping)
end

function month_of_year_and_start(time)
    year = floor(Int, time / SECONDS_PER_YEAR)
    second_of_year = time - year * SECONDS_PER_YEAR
    month = clamp(searchsortedlast(CUMULATIVE_MONTH_SECONDS, second_of_year + eps(second_of_year)), 1, 12)
    month_start = year * SECONDS_PER_YEAR + CUMULATIVE_MONTH_SECONDS[month]
    month_stop = year * SECONDS_PER_YEAR + CUMULATIVE_MONTH_SECONDS[month + 1]
    return year, month, month_start, month_stop
end

function split_interval_by_month(interval_start, interval_end)
    interval_end <= interval_start && return Tuple{Int, Int, Float64, Float64}[]
    pieces = Tuple{Int, Int, Float64, Float64}[]
    current = Float64(interval_start)
    stop = Float64(interval_end)
    while current < stop
        year, month, _, month_stop = month_of_year_and_start(current)
        piece_stop = min(stop, month_stop)
        push!(pieces, (year, month, current, piece_stop))
        current = piece_stop
    end
    return pieces
end

function read_serialized_grid(file)
    haskey(file, "serialized/grid") && return file["serialized/grid"]
    haskey(file, "serialized") || return nothing
    for key in sort!(String.(collect(keys(file["serialized"]))))
        startswith(key, "grid") && return file["serialized/$key"]
    end
    return nothing
end

function load_grid_from_output_file(filepath)
    return with_logger(NullLogger()) do
        JLD2.jldopen(filepath, "r") do file
            read_serialized_grid(file)
        end
    end
end

function log_progress(tag, current, total)
    width = 24
    fraction = current / total
    filled = clamp(floor(Int, width * fraction), 0, width)
    bar = "[" * repeat("=", filled) * repeat(".", width - filled) * "]"
    @info "Progress." tag current total percent = round(100 * fraction; digits = 1) bar
    return nothing
end

function bottom_height_matrix(grid)
    isnothing(grid) && return nothing
    hasproperty(grid, :immersed_boundary) || return nothing
    immersed_boundary = getproperty(grid, :immersed_boundary)
    hasproperty(immersed_boundary, :bottom_height) || return nothing
    bottom_height = getproperty(immersed_boundary, :bottom_height)
    hasproperty(bottom_height, :data) || return nothing
    return physical_matrix(getproperty(bottom_height, :data), underlying_grid(grid); T = Float32)
end

function apply_plot_mask(source, mask)
    destination = Float32.(source)
    isnothing(mask) && return destination
    @inbounds for i in eachindex(destination, mask)
        mask[i] || (destination[i] = NaN32)
    end
    return destination
end

function apply_plot_mask!(destination, source, mask)
    if isnothing(mask)
        copyto!(destination, source)
    else
        @inbounds for i in eachindex(destination, source, mask)
            destination[i] = mask[i] ? Float32(source[i]) : NaN32
        end
    end
    return destination
end

function build_error_frame!(destination, model, reference, mask)
    @inbounds for i in eachindex(destination, model, reference)
        valid = isfinite(model[i]) && isfinite(reference[i]) && (isnothing(mask) || mask[i])
        destination[i] = valid ? Float32(model[i] - reference[i]) : NaN32
    end
    return destination
end

function build_error_frame(model, reference, mask)
    destination = Matrix{Float32}(undef, size(model))
    return build_error_frame!(destination, model, reference, mask)
end

function constant_model_dt_framerate(times; fallback = VIDEO_FRAMERATE, sim_years_per_second = VIDEO_SIM_YEARS_PER_SECOND)
    length(times) > 1 || return Int(round(fallback))
    deltas = [times[i + 1] - times[i] for i in 1:length(times)-1 if times[i + 1] > times[i]]
    isempty(deltas) && return Int(round(fallback))
    fps = sim_years_per_second * SECONDS_PER_YEAR / median(deltas)
    return round(Int, clamp(fps, MIN_VIDEO_FRAMERATE, MAX_VIDEO_FRAMERATE))
end

function record_video_in_chunks(render_frame!, figure, outname, frames, framerate; chunk_size = VIDEO_CHUNK_SIZE, tag = "video", chunk_cleanup = nothing)
    isempty(frames) && error("No video frames were provided.")
    mkpath(dirname(outname))
    ffmpeg = Sys.which("ffmpeg")
    fallback = "/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/13.3.0/FFmpeg/7.0.2/bin/ffmpeg"
    isnothing(ffmpeg) && isfile(fallback) && (ffmpeg = fallback)
    isnothing(ffmpeg) && error("Could not find ffmpeg.")
    chunk_dir = mktempdir(dirname(outname); prefix = basename(outname) * "_chunks_")
    chunk_paths = String[]
    try
        for (chunk_index, first_frame) in enumerate(1:chunk_size:length(frames))
            last_frame = min(length(frames), first_frame + chunk_size - 1)
            chunk_frames = collect(frames[first_frame:last_frame])
            chunk_path = joinpath(chunk_dir, "chunk_" * lpad(string(chunk_index), 4, "0") * ".mp4")
            push!(chunk_paths, chunk_path)
            @info "Recording video chunk." tag chunk_index first_frame last_frame
            record(figure, chunk_path, chunk_frames; framerate) do frame
                render_frame!(frame)
            end
            isnothing(chunk_cleanup) || chunk_cleanup()
            GC.gc(false)
        end
        concat_path = joinpath(chunk_dir, "concat.txt")
        open(concat_path, "w") do io
            quote_character = Char(39)
            for chunk_path in chunk_paths
                println(io, "file " * quote_character * abspath(chunk_path) * quote_character)
            end
        end
        run(`$ffmpeg -y -f concat -safe 0 -i $concat_path -c copy $outname`)
    finally
        rm(chunk_dir; recursive = true, force = true)
    end
    return outname
end
