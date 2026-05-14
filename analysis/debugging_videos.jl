using CairoMakie
using Glob
using JLD2
using Oceananigans
using Printf

with_trailing_slash(path) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator

const OUTPUT_PATH = with_trailing_slash(expanduser(get(ENV, "OUTPUT_PATH", "/home/tsohail/uom/ocean-ensembles/outputs/")))
const FIGDIR = with_trailing_slash(expanduser(get(ENV, "FIGDIR", "/home/tsohail/uom/ocean-ensembles/figures/")))
const DEBUG_VIDEO_DIR = joinpath(FIGDIR, "debugging_videos")
const RESOLUTION = "sxtdeg"
const SECONDS_PER_DAY = 24 * 60 * 60
const VIDEO_FRAMERATE = parse(Int, get(ENV, "VIDEO_FRAMERATE", "12"))
const STD_COLOR_FACTOR = parse(Float64, get(ENV, "STD_COLOR_FACTOR", "4"))
const DIVERGING_COLORMAP = :bwr
const AUTO_DIVERGING_CENTER_TOLERANCE = parse(Float64, get(ENV, "AUTO_DIVERGING_CENTER_TOLERANCE", "0.1"))

const SUBSURFACE_PREFIX = "global_diagnostic_k74_fields_$(RESOLUTION)_RYF_run"
const SURFACE_PREFIX = "global_diagnostic_surface_fields_$(RESOLUTION)_RYF_run"

const COLOR_SETTINGS = Dict(
    "T" => (:thermal, nothing),
    "S" => (:viridis, nothing),
    "u" => (DIVERGING_COLORMAP, nothing),
    "v" => (DIVERGING_COLORMAP, nothing),
    "w" => (DIVERGING_COLORMAP, nothing),
    "e" => (DIVERGING_COLORMAP, nothing),
    "speed" => (:speed, nothing),
    "surface_height" => (DIVERGING_COLORMAP, nothing),
    "net_ocean_flux_T" => (DIVERGING_COLORMAP, nothing),
    "net_ocean_flux_S" => (DIVERGING_COLORMAP, nothing),
    "net_ocean_flux_u" => (DIVERGING_COLORMAP, nothing),
    "net_ocean_flux_v" => (DIVERGING_COLORMAP, nothing),
    "atmosphere_ocean_sensible_heat" => (DIVERGING_COLORMAP, nothing),
    "atmosphere_ocean_latent_heat" => (DIVERGING_COLORMAP, nothing),
    "atmosphere_ocean_water_vapor" => (DIVERGING_COLORMAP, nothing),
    "atmosphere_ocean_x_momentum" => (DIVERGING_COLORMAP, nothing),
    "atmosphere_ocean_y_momentum" => (DIVERGING_COLORMAP, nothing),
    "sea_ice_ocean_interface_heat" => (DIVERGING_COLORMAP, nothing),
    "sea_ice_ocean_frazil_heat" => (DIVERGING_COLORMAP, nothing),
    "sea_ice_ocean_salt" => (DIVERGING_COLORMAP, nothing),
    "sea_ice_ocean_x_momentum" => (DIVERGING_COLORMAP, nothing),
    "sea_ice_ocean_y_momentum" => (DIVERGING_COLORMAP, nothing),
    "sea_ice_thickness" => (:ice, nothing),
    "sea_ice_concentration" => (:ice, (0f0, 1f0)),
    "sea_ice_top_surface_temperature" => (:thermal, nothing),
    "sea_ice_u" => (DIVERGING_COLORMAP, nothing),
    "sea_ice_v" => (DIVERGING_COLORMAP, nothing),
    "ocean_radiation_upwelling_longwave" => (:viridis, nothing),
    "ocean_radiation_downwelling_longwave" => (:viridis, nothing),
    "ocean_radiation_downwelling_shortwave" => (:viridis, nothing),
)

const NON_DIVERGING_VARIABLES = Set([
    "T",
    "S",
    "speed",
    "sea_ice_thickness",
    "sea_ice_concentration",
    "sea_ice_top_surface_temperature",
    "ocean_radiation_upwelling_longwave",
    "ocean_radiation_downwelling_longwave",
    "ocean_radiation_downwelling_shortwave",
])

@inline function run_id(path::AbstractString)
    m = match(r"run(\d+)", basename(path))
    return m === nothing ? -1 : parse(Int, m.captures[1])
end

function parse_run_filter()
    isempty(ARGS) && return nothing
    return Set(parse.(Int, ARGS))
end

function diagnostic_files(prefix::String; run_filter = nothing)
    files = glob(prefix * "*.jld2", OUTPUT_PATH)
    files = filter(files) do file
        !occursin("_rank", file) && run_id(file) >= 0 &&
            (run_filter === nothing || run_id(file) in run_filter)
    end
    sort!(files; by = run_id)
    return files
end

function timeseries_variables(file::AbstractString)
    jldopen(file, "r") do f
        haskey(f, "timeseries") || return String[]
        vars = String[]
        for key in keys(f["timeseries"])
            key == "t" && continue
            haskey(f, "timeseries/$key") && push!(vars, String(key))
        end
        sort!(vars)
        return vars
    end
end

function numeric_timeseries_keys(group)
    keys_int = Int[]

    for key in keys(group)
        parsed = tryparse(Int, String(key))
        parsed === nothing || push!(keys_int, parsed)
    end

    sort!(keys_int)
    return keys_int
end

function timeseries_keys(file::AbstractString, var::String)
    jldopen(file, "r") do f
        haskey(f, "timeseries/t") || return Int[]
        haskey(f, "timeseries/$var") || return Int[]
        tkeys = Set(numeric_timeseries_keys(f["timeseries/t"]))
        vkeys = Set(numeric_timeseries_keys(f["timeseries/$var"]))
        return sort!(collect(intersect(tkeys, vkeys)))
    end
end

function speed_keys(file::AbstractString)
    jldopen(file, "r") do f
        all(haskey(f, key) for key in ("timeseries/t", "timeseries/u", "timeseries/v")) || return Int[]
        tkeys = Set(numeric_timeseries_keys(f["timeseries/t"]))
        ukeys = Set(numeric_timeseries_keys(f["timeseries/u"]))
        vkeys = Set(numeric_timeseries_keys(f["timeseries/v"]))
        return sort!(collect(intersect(tkeys, ukeys, vkeys)))
    end
end

function extract_2d_f32(raw)
    A = Array(raw)
    if ndims(A) == 2
        return Float32.(A)
    elseif ndims(A) == 3
        return Float32.(A[:, :, 1])
    else
        error("Expected a 2D field or a 3D singleton slice; got size $(size(A)).")
    end
end

function load_frame(file::AbstractString, var::String, key::Int)
    jldopen(file, "r") do f
        return extract_2d_f32(f["timeseries/$var/$key"])
    end
end

function load_speed_frame(file::AbstractString, key::Int)
    jldopen(file, "r") do f
        u = extract_2d_f32(f["timeseries/u/$key"])
        v = extract_2d_f32(f["timeseries/v/$key"])
        speed = similar(u, Float32)
        @inbounds for i in eachindex(speed, u, v)
            speed[i] = sqrt(u[i]^2 + v[i]^2)
        end
        return speed
    end
end

function load_time_days(file::AbstractString, key::Int)
    jldopen(file, "r") do f
        return Float64(f["timeseries/t/$key"]) / SECONDS_PER_DAY
    end
end

function load_serialized_grid(file::AbstractString)
    jldopen(file, "r") do f
        if haskey(f, "serialized/grid")
            return f["serialized/grid"]
        elseif haskey(f, "serialized/grid_1")
            return f["serialized/grid_1"]
        else
            return nothing
        end
    end
end

function lon_lat_arrays(file::AbstractString)
    grid = load_serialized_grid(file)
    grid === nothing && return nothing

    try
        lon = Array(λnodes(grid, Center(), Center(), Center(); with_halos = false))
        lat = Array(φnodes(grid, Center(), Center(), Center(); with_halos = false))

        ndims(lon) == 3 && (lon = lon[:, :, 1])
        ndims(lat) == 3 && (lat = lat[:, :, 1])

        return Float32.(lon), Float32.(lat)
    catch err
        @warn "Could not load lon/lat arrays for NaN annotation." file exception = (err, catch_backtrace())
        return nothing
    end
end

function lon_lat_at(file::AbstractString, index::CartesianIndex)
    lon_lat = lon_lat_arrays(file)
    lon_lat === nothing && return nothing

    lon, lat = lon_lat
    i, j = Tuple(index)
    ii = clamp(i, axes(lon, 1))
    jj = clamp(j, axes(lon, 2))

    return (lon = lon[ii, jj], lat = lat[ii, jj])
end

function first_nan_annotation(frames::Vector{Tuple{String, Int}}, var::String; is_speed::Bool = false)
    for (frame_index, (file, key)) in enumerate(frames)
        A = is_speed ? load_speed_frame(file, key) : load_frame(file, var, key)
        indices = findall(isnan, A)
        isempty(indices) && continue

        index = first(indices)
        i, j = Tuple(index)
        lon_lat = lon_lat_at(file, index)
        time_days = load_time_days(file, key)
        nan_count = length(indices)

        label = if lon_lat === nothing
            @sprintf("first NaNs: %d cells\nexample (i, j)=(%d, %d)", nan_count, i, j)
        else
            @sprintf("first NaNs: %d cells\nexample lat=%.3f\nexample lon=%.3f", nan_count, lon_lat.lat, lon_lat.lon)
        end

        return (; frame_index, index, nan_count, label, time_days)
    end

    return nothing
end

mutable struct RunningMoments
    n::Int
    mean::Float64
    m2::Float64
    minimum::Float64
    maximum::Float64
end

RunningMoments() = RunningMoments(0, 0.0, 0.0, Inf, -Inf)

function update_moments!(moments::RunningMoments, A)
    @inbounds for value in A
        isfinite(value) || continue
        value64 = Float64(value)
        moments.n += 1
        moments.minimum = min(moments.minimum, value64)
        moments.maximum = max(moments.maximum, value64)
        δ = value64 - moments.mean
        moments.mean += δ / moments.n
        moments.m2 += δ * (value64 - moments.mean)
    end

    return moments
end

function moments_std(moments::RunningMoments)
    moments.n == 0 && error("No finite values found.")
    return moments.n > 1 ? sqrt(moments.m2 / (moments.n - 1)) : 0.0
end

function widen_degenerate_range(lo::Real, hi::Real)
    if lo == hi
        delta = max(abs(Float32(lo)), 1f0) * 1f-6
        return (Float32(lo) - delta, Float32(hi) + delta)
    end

    return (Float32(lo), Float32(hi))
end

function zero_centered_range(colorrange)
    lo, hi = colorrange
    lo < 0 < hi || return false
    return abs(lo + hi) <= AUTO_DIVERGING_CENTER_TOLERANCE * (hi - lo)
end

function symmetric_zero_range(colorrange)
    lo, hi = colorrange
    limit = max(abs(lo), abs(hi), eps(Float32))
    return (-limit, limit)
end

function auto_diverging_candidate(var::String; is_speed::Bool = false)
    is_speed && return false
    var in NON_DIVERGING_VARIABLES && return false

    return var in ("u", "v", "w", "e", "surface_height") ||
           endswith(var, "_u") ||
           endswith(var, "_v") ||
           occursin("momentum", var) ||
           occursin("flux", var)
end

function variable_colorrange(files::Vector{String}, var::String; is_speed::Bool = false)
    override = get(COLOR_SETTINGS, var, (:viridis, nothing))[2]
    override !== nothing && return override

    colormap = get(COLOR_SETTINGS, var, (:viridis, nothing))[1]
    moments = RunningMoments()

    for file in files
        keys = is_speed ? speed_keys(file) : timeseries_keys(file, var)
        for key in keys
            A = is_speed ? load_speed_frame(file, key) : load_frame(file, var, key)
            update_moments!(moments, A)
        end
    end

    σ = moments_std(moments)
    limit = STD_COLOR_FACTOR * σ

    if colormap == DIVERGING_COLORMAP
        limit = max(limit, eps(Float32))
        return (Float32(-limit), Float32(limit))
    end

    lo = moments.mean - limit
    hi = moments.mean + limit
    lo = max(lo, moments.minimum)
    hi = min(hi, moments.maximum)
    return widen_degenerate_range(lo, hi)
end

function variable_color_settings(files::Vector{String}, var::String; is_speed::Bool = false)
    preferred_colormap = get(COLOR_SETTINGS, var, (:viridis, nothing))[1]
    colorrange = variable_colorrange(files, var; is_speed)

    if preferred_colormap == DIVERGING_COLORMAP
        return DIVERGING_COLORMAP, symmetric_zero_range(colorrange)
    elseif auto_diverging_candidate(var; is_speed) && zero_centered_range(colorrange)
        return DIVERGING_COLORMAP, symmetric_zero_range(colorrange)
    else
        return preferred_colormap, colorrange
    end
end

function frames_for_variable(files::Vector{String}, var::String; is_speed::Bool = false)
    frames = Tuple{String, Int}[]
    for file in files
        keys = is_speed ? speed_keys(file) : timeseries_keys(file, var)
        append!(frames, [(file, key) for key in keys])
    end

    sort!(frames; by = frame -> (load_time_days(frame[1], frame[2]), run_id(frame[1]), frame[2]))
    return frames
end

function log_record_progress(var::String, frame::Int, nframes::Int)
    width = 24
    fraction = frame / nframes
    filled = clamp(floor(Int, width * fraction), 0, width)
    bar = "[" * repeat("=", filled) * repeat(".", width - filled) * "]"
    @info "Recording progress." variable = var frame nframes percent = round(100 * fraction; digits = 1) bar
end

function video_filename(source::String, var::String)
    return joinpath(DEBUG_VIDEO_DIR, "$(source)_$(var)_$(RESOLUTION)_debug.mp4")
end

function make_variable_video(files::Vector{String}, var::String, source::String; is_speed::Bool = false)
    frames = frames_for_variable(files, var; is_speed)
    isempty(frames) && return nothing

    colormap, colorrange = variable_color_settings(files, var; is_speed)
    first_file, first_key = first(frames)
    A0 = is_speed ? load_speed_frame(first_file, first_key) : load_frame(first_file, var, first_key)
    data = Observable(A0)
    nan_annotation = first_nan_annotation(frames, var; is_speed)

    fig = Figure(size = (1000, 720))
    ax = Axis(fig[1, 1], title = "$var | time = $(round(load_time_days(first_file, first_key), digits = 3)) days")
    hm = heatmap!(ax, data; colormap, colorrange)
    nan_x = Observable(Float32[])
    nan_y = Observable(Float32[])
    nan_text = Observable(String[])
    nan_text_x = Observable(Float32[])
    nan_text_y = Observable(Float32[])
    scatter!(ax, nan_x, nan_y; color = :red, markersize = 14)
    text!(ax, nan_text_x, nan_text_y; text = nan_text, color = :red, fontsize = 16, align = (:right, :top))
    Colorbar(fig[1, 2], hm; label = var)

    outname = video_filename(source, var)
    nframes = length(frames)
    progress_step = max(1, cld(nframes, 20))

    @info "Recording debug animation." source variable = var nframes colorrange std_factor = STD_COLOR_FACTOR outname
    record(fig, outname, 1:nframes; framerate = VIDEO_FRAMERATE) do frame
        file, key = frames[frame]
        time_days = load_time_days(file, key)
        ax.title = "$var | time = $(round(time_days, digits = 3)) days"
        A = is_speed ? load_speed_frame(file, key) : load_frame(file, var, key)
        data[] = A

        if nan_annotation !== nothing && frame >= nan_annotation.frame_index
            i, j = Tuple(nan_annotation.index)
            nan_x[] = Float32[i]
            nan_y[] = Float32[j]
            nan_text[] = [nan_annotation.label]
            nan_text_x[] = Float32[i]
            nan_text_y[] = Float32[j]
        else
            nan_x[] = Float32[]
            nan_y[] = Float32[]
            nan_text[] = String[]
            nan_text_x[] = Float32[]
            nan_text_y[] = Float32[]
        end

        if frame == 1 || frame == nframes || frame % progress_step == 0
            log_record_progress(var, frame, nframes)
        end
    end

    @info "Saved debug animation." variable = var outname
    return outname
end

function make_debugging_videos()
    mkpath(DEBUG_VIDEO_DIR)
    run_filter = parse_run_filter()

    subsurface_files = diagnostic_files(SUBSURFACE_PREFIX; run_filter)
    surface_files = diagnostic_files(SURFACE_PREFIX; run_filter)

    isempty(subsurface_files) && @warn "No diagnostic subsurface files found." pattern = SUBSURFACE_PREFIX * "*.jld2"
    isempty(surface_files) && @warn "No diagnostic surface files found." pattern = SURFACE_PREFIX * "*.jld2"

    outputs = String[]

    if !isempty(subsurface_files)
        vars = timeseries_variables(first(subsurface_files))
        for var in vars
            out = make_variable_video(subsurface_files, var, "diagnostic_subsurface")
            out === nothing || push!(outputs, out)
        end

        if all(var in vars for var in ("u", "v"))
            out = make_variable_video(subsurface_files, "speed", "diagnostic_subsurface"; is_speed = true)
            out === nothing || push!(outputs, out)
        else
            @warn "Skipping speed: diagnostic subsurface files do not contain both u and v."
        end
    end

    if !isempty(surface_files)
        vars = timeseries_variables(first(surface_files))
        for var in vars
            out = make_variable_video(surface_files, var, "diagnostic_surface")
            out === nothing || push!(outputs, out)
        end
    end

    @info "Finished debugging videos." count = length(outputs) output_dir = DEBUG_VIDEO_DIR
    return outputs
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_debugging_videos()
end
