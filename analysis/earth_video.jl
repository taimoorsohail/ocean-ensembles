using CairoMakie
using Glob
using JLD2
using NumericalEarth: EarthSystemModels, Oceans
using Oceananigans
using Oceananigans.Fields: location, interior

const OUTPUT_PATH = normpath(joinpath(@__DIR__, "..", "outputs", "saved"))
const FIGDIR = normpath(joinpath(@__DIR__, "..", "figures"))
const RESOLUTION = "sxtdeg"
const DEPTH_LEVEL = 75
const DISPLAY_YEARS_PER_VARIABLE = 2.0
const CAMERA_SPIN_YEARS = 4.0
const VIDEO_FRAMERATE = 12
const VIDEO_OUTNAME = joinpath(FIGDIR, "all_vars_earth_vid.mp4")
const SECONDS_PER_YEAR = 365 * 24 * 60 * 60
const LATITUDE_SWEEP_MAX_DEG = 50.0
const STATIC_AZIMUTH_DEG = -35.0
const SURFACE_COLOR_SIGMA_MULTIPLE = 3.0f0
const DEPTH_COLOR_SIGMA_MULTIPLE = 1.0f0
const MAX_COLOR_SAMPLE_FRAMES = 120
const GC_INTERVAL = 24
const EARTH_UNDERLAY_COLOR = RGBf(0.80, 0.82, 0.78)
const VIDEO_VARS = ["heat_flux", "fw_flux", "surface_height", "T", "S", "speed", "w"]

const VAR_LABELS = Dict(
    "heat_flux" => "Heat Flux (W m⁻²)",
    "fw_flux" => "Freshwater Flux (kg m⁻² s⁻¹)",
    "surface_height" => "SSH (m)",
    "T" => "Temperature (°C)",
    "S" => "Salinity (g kg⁻¹)",
    "speed" => "Horizontal Speed (m s⁻¹)",
    "w" => "Vertical Velocity (m s⁻¹)")

const ocean_eos = Oceans.TEOS10EquationOfState()
const ρ₀ = isdefined(Oceananigans, :reference_density) ?
           Oceananigans.reference_density(ocean_eos) :
           EarthSystemModels.reference_density(ocean_eos)
const cₚ = isdefined(Oceananigans, :heat_capacity) ?
           Oceananigans.heat_capacity(ocean_eos) :
           EarthSystemModels.heat_capacity(ocean_eos)

const SPEED_WORKSPACE_CACHE = IdDict{Any, Any}()

struct FrameRef
    file::String
    key::Int
    time::Float64
    run::Int
end

mutable struct VariableSource
    var::String
    kind::Symbol
    refs::Vector{FrameRef}
    current_file::String
    file_handle::Any
end

VariableSource(var::String, kind::Symbol, refs::Vector{FrameRef}) =
    VariableSource(var, kind, refs, "", nothing)

struct RenderFrame
    var::String
    ref::FrameRef
    video_year::Float64
end

struct OHCSeries
    time_years::Vector{Float64}
    ohc_anomaly::Vector{Float64}
end

run_id(path::AbstractString) = begin
    m = match(r"run(\d+)", basename(path))
    m === nothing ? -1 : parse(Int, m.captures[1])
end

function source_files(kind::Symbol)
    pattern = kind == :surface ?
              "combined_global_surface_fluxes_$(RESOLUTION)_RYF_run*.jld2" :
              "combined_global_$(DEPTH_LEVEL)_fields_$(RESOLUTION)_RYF_run*.jld2"
    files = glob(pattern, OUTPUT_PATH)
    files = filter(f -> !occursin("_rank", basename(f)) && run_id(f) >= 0, files)
    sort!(files; by = run_id)
    return files
end

source_kind(var::String) = var in ("heat_flux", "fw_flux", "surface_height") ? :surface : :depth
required_keys(var::String) = var == "speed" ? ("u", "v") : (var,)

function read_serialized_grid(file)
    haskey(file, "serialized/grid") && return file["serialized/grid"]
    haskey(file, "serialized") || return nothing

    for key in sort!(String.(collect(keys(file["serialized"]))))
        startswith(key, "grid") || continue
        return file["serialized/$key"]
    end

    return nothing
end

function load_serialized_grid(files::Vector{String})
    for file in files
        grid = jldopen(file, "r") do data
            read_serialized_grid(data)
        end
        isnothing(grid) || return grid
    end

    error("No serialized grid found in candidate files.")
end

function collect_frame_refs(files::Vector{String}, var::String)
    refs_by_time = Dict{Float64, FrameRef}()
    needed = required_keys(var)

    for file in files
        run = run_id(file)
        jldopen(file, "r") do data
            haskey(data, "timeseries/t") || return
            available = Set(String.(collect(keys(data["timeseries"]))))
            all(key -> key in available, needed) || return

            keys_sorted = sort(parse.(Int, collect(keys(data["timeseries/t"]))))
            for key in keys_sorted
                t = Float64(data["timeseries/t/$key"])
                candidate = FrameRef(file, key, t, run)
                existing = get(refs_by_time, t, nothing)
                if isnothing(existing) || run > existing.run || (run == existing.run && key >= existing.key)
                    refs_by_time[t] = candidate
                end
            end
        end
    end

    refs = collect(values(refs_by_time))
    sort!(refs; by = ref -> ref.time)
    isempty(refs) && error("No frames found for variable $var.")
    return refs
end

function build_source(var::String)
    kind = source_kind(var)
    files = source_files(kind)
    isempty(files) && error("No files found for source kind $kind.")
    refs = collect_frame_refs(files, var)
    return VariableSource(var, kind, refs)
end

function close_source_file!(source::VariableSource)
    if source.file_handle !== nothing
        close(source.file_handle)
        source.file_handle = nothing
    end
    source.current_file = ""
    return nothing
end

function ensure_source_file!(source::VariableSource, filepath::String)
    source.current_file == filepath && return source.file_handle
    close_source_file!(source)
    source.file_handle = jldopen(filepath, "r")
    source.current_file = filepath
    return source.file_handle
end

function extract_2d(raw)
    if ndims(raw) == 2
        return raw
    elseif ndims(raw) == 3
        return view(raw, :, :, 1)
    end
    error("Unsupported array rank.")
end

@inline surface_matrix_3d(A::AbstractMatrix) = reshape(A, size(A, 1), size(A, 2), 1)

function ensure_speed_workspace!(source::VariableSource, file)
    haskey(SPEED_WORKSPACE_CACHE, source) && return SPEED_WORKSPACE_CACHE[source]

    grid = read_serialized_grid(file)
    isnothing(grid) && error("No serialized grid found while computing speed.")

    workspace = let
        ufield = XFaceField(grid)
        vfield = YFaceField(grid)
        speed_field = @at (Center, Center, Nothing) sqrt(ufield^2 + vfield^2) |> Field
        (; ufield, vfield, speed_field)
    end

    SPEED_WORKSPACE_CACHE[source] = workspace
    return workspace
end

function load_frame(source::VariableSource, ref::FrameRef)
    file = ensure_source_file!(source, ref.file)

    if source.var == "speed"
        u = Float32.(extract_2d(file["timeseries/u/$(ref.key)"]))
        v = Float32.(extract_2d(file["timeseries/v/$(ref.key)"]))
        workspace = ensure_speed_workspace!(source, file)
        set!(workspace.ufield, surface_matrix_3d(u))
        set!(workspace.vfield, surface_matrix_3d(v))
        Oceananigans.fill_halo_regions!(workspace.ufield)
        Oceananigans.fill_halo_regions!(workspace.vfield)
        compute!(workspace.speed_field)
        return Float32.(Array(interior(workspace.speed_field)[:, :, 1]))
    end

    return Float32.(extract_2d(file["timeseries/$(source.var)/$(ref.key)"]))
end

function final_window_refs(refs::Vector{FrameRef}; duration_years::Real = DISPLAY_YEARS_PER_VARIABLE)
    last_time = refs[end].time
    first_time = last_time - duration_years * SECONDS_PER_YEAR
    window_refs = filter(ref -> ref.time >= first_time, refs)
    isempty(window_refs) && return refs
    return window_refs
end

function sampled_reference_indices(n::Int; max_samples::Int = MAX_COLOR_SAMPLE_FRAMES)
    n <= 0 && return Int[]
    n <= max_samples && return collect(1:n)
    return unique(round.(Int, range(1, n; length = max_samples)))
end

function sample_depth_temperature_limits(source::VariableSource)
    frame = load_frame(source, source.refs[end])
    n = 0
    mean_value = 0.0
    m2 = 0.0

    @info "Computing Earth-video temperature style from last frame" variable = source.var frame_index = length(source.refs)

    for value in frame
        isfinite(value) || continue
        value64 = Float64(value)
        n += 1
        δ = value64 - mean_value
        mean_value += δ / n
        m2 += δ * (value64 - mean_value)
    end

    if n <= 1
        center = Float32(mean_value)
        pad = 1f-6
        return :thermal, (center - pad, center + pad)
    end

    std_value = sqrt(m2 / (n - 1))
    halfwidth = max(Float32(DEPTH_COLOR_SIGMA_MULTIPLE * std_value), 1f-6)
    center = Float32(mean_value)
    return :thermal, (center - halfwidth, center + halfwidth)
end

function sample_surface_balance_limits(source::VariableSource)
    frame = load_frame(source, source.refs[end])
    n = 0
    mean_value = 0.0
    m2 = 0.0

    @info "Computing Earth-video surface style from last frame" variable = source.var frame_index = length(source.refs)

    for value in frame
        isfinite(value) || continue
        value64 = Float64(value)
        n += 1
        δ = value64 - mean_value
        mean_value += δ / n
        m2 += δ * (value64 - mean_value)
    end

    if n <= 1
        return :balance, (-1f0, 1f0)
    end

    std_value = sqrt(m2 / (n - 1))
    kσ = max(Float32(SURFACE_COLOR_SIGMA_MULTIPLE * std_value), 1f-6)
    return :balance, (-kσ, kσ)
end

function variable_style(var::String, source::VariableSource)
    if var == "T"
        return sample_depth_temperature_limits(source)
    elseif var == "S"
        return :haline, (34.8f0, 37f0)
    elseif var == "speed"
        return :speed, (0f0, 0.7f0)
    elseif var == "w"
        return :balance, (-2f-5, 2f-5)
    elseif var in ("surface_height", "heat_flux", "fw_flux")
        return sample_surface_balance_limits(source)
    end

    return :viridis, (0f0, 1f0)
end

function build_render_plan(vars::Vector{String}, sources::Dict{String, VariableSource})
    plan = RenderFrame[]

    for (segment_index, var) in enumerate(vars)
        refs = final_window_refs(sources[var].refs)
        first_time = refs[1].time
        last_time = refs[end].time
        span = max(last_time - first_time, 1.0)
        segment_start_year = (segment_index - 1) * DISPLAY_YEARS_PER_VARIABLE

        for ref in refs
            progress = (ref.time - first_time) / span
            video_year = segment_start_year + DISPLAY_YEARS_PER_VARIABLE * progress
            push!(plan, RenderFrame(var, ref, video_year))
        end
    end

    isempty(plan) && error("Render plan is empty.")
    return plan
end

function load_ohc_series()
    files = glob("combined_global_*tot*$(RESOLUTION)*_RYF_run*.jld2", OUTPUT_PATH)
    files = filter(f -> !occursin("_rank", basename(f)), files)
    sort!(files; by = run_id)
    isempty(files) && error("No combined integral files found for OHC inset.")

    integral_by_time = Dict{Float64, NamedTuple{(:run, :T), Tuple{Int, Float64}}}()

    for file in files
        run = run_id(file)
        jldopen(file, "r") do data
            haskey(data, "timeseries/t") || return
            keys_sorted = sort(parse.(Int, collect(keys(data["timeseries/t"]))))
            for key in keys_sorted
                t = Float64(data["timeseries/t/$key"])
                T = Float64(data["timeseries/T_totintegral/$key"][1, 1, 1])
                existing = get(integral_by_time, t, nothing)
                if isnothing(existing) || run >= existing.run
                    integral_by_time[t] = (run = run, T = T)
                end
            end
        end
    end

    times = sort(collect(keys(integral_by_time)))
    isempty(times) && error("No OHC times found.")
    ohc = [ρ₀ * cₚ * integral_by_time[t].T for t in times]
    ohc_anomaly = ohc .- ohc[1]
    return OHCSeries(times ./ SECONDS_PER_YEAR, ohc_anomaly)
end

function interpolate_series_value(xs::Vector{Float64}, ys::Vector{Float64}, x::Float64)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]

    upper = searchsortedfirst(xs, x)
    lower = max(1, upper - 1)
    x0, x1 = xs[lower], xs[upper]
    y0, y1 = ys[lower], ys[upper]
    x1 == x0 && return y0

    α = (x - x0) / (x1 - x0)
    return (1 - α) * y0 + α * y1
end

function spherical_coordinates_viz(λ, φ, r = 1)
    λ_rad = deg2rad.(λ)
    φ_rad = deg2rad.(φ)

    x = @. r * cos(φ_rad) * cos(λ_rad)
    y = @. r * cos(φ_rad) * sin(λ_rad)
    z = @. r * sin(φ_rad)

    return x, y, z
end

function earth_underlay()
    n = 1024 ÷ 4
    lat = reverse(LinRange(-π / 2, π / 2, n))
    lon = LinRange(-π, π, 2 * n)
    r = 1.0
    r_offset = -0.02

    x = [(r + r_offset) * cos(latv) * cos(lonv) for latv in lat, lonv in lon]
    y = [(r + r_offset) * cos(latv) * sin(lonv) for latv in lat, lonv in lon]
    z = [(r + r_offset) * sin(latv) for latv in lat, lonv in lon]

    return x, y, z
end

function maybe_bottom_height_matrix(file)
    jldopen(file, "r") do data
        grid = read_serialized_grid(data)
        isnothing(grid) && return nothing
        hasproperty(grid, :immersed_boundary) || return nothing
        immersed_boundary = getproperty(grid, :immersed_boundary)
        hasproperty(immersed_boundary, :bottom_height) || return nothing
        return Float32.(Array(interior(getproperty(immersed_boundary, :bottom_height), :, :, 1)))
    end
end

function apply_plot_mask!(dest::Matrix{Float32}, src::Matrix{Float32}, plot_mask::Union{Nothing, AbstractMatrix{Bool}})
    size(dest) == size(src) || error("Destination/src size mismatch.")

    if isnothing(plot_mask)
        copyto!(dest, src)
        return dest
    end

    @inbounds for i in eachindex(dest, src, plot_mask)
        dest[i] = plot_mask[i] ? src[i] : NaN32
    end

    return dest
end

function camera_angles_deg(video_year::Real, total_video_years::Real)
    azimuth_deg = STATIC_AZIMUTH_DEG + 360 * (video_year / CAMERA_SPIN_YEARS)
    azimuth_deg = mod(azimuth_deg + 180, 360) - 180
    phase = clamp(video_year / max(Float64(total_video_years), eps(Float64)), 0.0, 1.0)
    elevation_deg = LATITUDE_SWEEP_MAX_DEG * cos(pi * phase)
    return elevation_deg, azimuth_deg
end

frame_png_path(frames_dir::AbstractString, frame_index::Int) =
    joinpath(frames_dir, "frame_" * lpad(frame_index, 6, '0') * ".png")

function assemble_earth_video_from_pngs(frames_dir::AbstractString,
                                        total_frames::Int;
                                        outname::String = VIDEO_OUTNAME,
                                        framerate::Int = VIDEO_FRAMERATE)
    ffmpeg = Sys.which("ffmpeg")
    isnothing(ffmpeg) && error("`ffmpeg` is required to assemble the Earth video.")

    missing = findfirst(frame_index -> !isfile(frame_png_path(frames_dir, frame_index)), 1:total_frames)
    isnothing(missing) || error("Missing PNG frame for Earth video assembly: " * frame_png_path(frames_dir, missing))

    input_pattern = joinpath(frames_dir, "frame_%06d.png")
    cmd = `$ffmpeg -y -framerate $framerate -i $input_pattern -pix_fmt yuv420p $outname`
    @info "Assembling Earth video from cached PNGs" frames_dir total_frames outname ffmpeg
    run(cmd)
    @info "Saved Earth video" outname frames = total_frames
    return outname
end

function build_sources_with_progress(vars::Vector{String})
    total = length(vars)
    sources = Dict{String, VariableSource}()

    for (index, var) in enumerate(vars)
        kind = source_kind(var)
        files = source_files(kind)
        isempty(files) && error("No files found for source kind $kind.")

        @info "Building Earth-video source ($index/$total)" variable = var kind files = length(files)
        refs = collect_frame_refs(files, var)
        sources[var] = VariableSource(var, kind, refs)
        @info "Built Earth-video source ($index/$total)" variable = var frames = length(refs) first_year = round(first(refs).time / SECONDS_PER_YEAR, digits = 2) last_year = round(last(refs).time / SECONDS_PER_YEAR, digits = 2)
    end

    return sources
end

function compute_styles_with_progress(vars::Vector{String}, sources::Dict{String, VariableSource})
    total = length(vars)
    styles = Dict{String, Tuple}()

    for (index, var) in enumerate(vars)
        @info "Computing Earth-video style ($index/$total)" variable = var frames = length(sources[var].refs)
        styles[var] = variable_style(var, sources[var])
        @info "Computed Earth-video style ($index/$total)" variable = var colormap = styles[var][1] colorrange = styles[var][2]
    end

    return styles
end

function make_earth_video(; outname::String = VIDEO_OUTNAME,
                           framerate::Int = VIDEO_FRAMERATE,
                           vars::Vector{String} = VIDEO_VARS,
                           frames_dir::String = joinpath(FIGDIR, "earth_video_frames"))
    mkpath(FIGDIR)
    mkpath(frames_dir)

    depth_files = source_files(:depth)
    isempty(depth_files) && error("No depth files found.")
    grid = load_serialized_grid(depth_files)
    center_field = CenterField(grid)
    ℓx, ℓy, ℓz = location(center_field)
    λ = λnodes(grid.underlying_grid, ℓx(), ℓy(), ℓz())
    φ = φnodes(grid.underlying_grid, ℓx(), ℓy(), ℓz())
    globe_x, globe_y, globe_z = spherical_coordinates_viz(λ, φ, 1)
    earth_x, earth_y, earth_z = earth_underlay()

    plot_mask = let bottom_height = maybe_bottom_height_matrix(depth_files[1])
        isnothing(bottom_height) ? nothing : bottom_height .< 0f0
    end

    @info "Building Earth-video sources" vars
    sources = build_sources_with_progress(vars)
    @info "Computing Earth-video styles"
    styles = compute_styles_with_progress(vars, sources)
    plan = build_render_plan(vars, sources)
    @info "Built Earth-video render plan" frames = length(plan) variables = length(vars)
    @info "Loading Earth-video OHC inset series"
    ohc_series = load_ohc_series()

    first_render = first(plan)
    first_frame = load_frame(sources[first_render.var], first_render.ref)
    masked_frame = similar(first_frame)
    apply_plot_mask!(masked_frame, first_frame, plot_mask)

    color_observable = Observable(masked_frame)
    cmap_observable = Observable(styles[first_render.var][1])
    clim_observable = Observable(styles[first_render.var][2])
    title_observable = Observable(VAR_LABELS[first_render.var])
    colorbar_label = Observable(VAR_LABELS[first_render.var])
    dot_x = Observable(first_render.ref.time / SECONDS_PER_YEAR)
    dot_y = Observable(interpolate_series_value(ohc_series.time_years, ohc_series.ohc_anomaly, dot_x[]))

    fig = Figure(size = (1100, 900))
    layout = fig[1, 1] = GridLayout()
    title = Label(layout[1, 1], title_observable, tellwidth = false)
    ax = Axis3(layout[2, 1], aspect = :data, viewmode = :fit)

    surface!(ax, earth_x, earth_y, earth_z;
             color = EARTH_UNDERLAY_COLOR,
             shading = NoShading,
             backlight = 1.5f0)

    hm = surface!(ax, globe_x, globe_y, globe_z;
                  color = color_observable,
                  colormap = cmap_observable,
                  colorrange = clim_observable,
                  nan_color = EARTH_UNDERLAY_COLOR)

    hidedecorations!(ax)
    hidespines!(ax)

    inset = Axis(layout[2, 1],
                 width = Relative(0.34),
                 height = Relative(0.22),
                 halign = 0.98,
                 valign = 0.02,
                 backgroundcolor = RGBAf(1, 1, 1, 0.86),
                 xlabel = "Year",
                 ylabel = "OHC (J)",
                 title = "OHC")
    lines!(inset, ohc_series.time_years, ohc_series.ohc_anomaly, color = :black, linewidth = 2)
    scatter!(inset, dot_x, dot_y, color = :red, markersize = 14)
    xlims!(inset, first(ohc_series.time_years), last(ohc_series.time_years))

    colorbar = Colorbar(layout[3, 1], hm, label = colorbar_label, vertical = false)
    resize_to_layout!(fig)

    total_frames = length(plan)
    total_video_years = length(vars) * DISPLAY_YEARS_PER_VARIABLE

    try
        for frame_index in 1:total_frames
            render = plan[frame_index]
            source = sources[render.var]
            png_path = frame_png_path(frames_dir, frame_index)

            if isfile(png_path)
                if frame_index == 1 || frame_index == total_frames || frame_index % 50 == 0
                    @info "Skipping existing Earth-video PNG" frame = frame_index total_frames path = png_path
                end
                continue
            end

            frame = load_frame(source, render.ref)
            apply_plot_mask!(color_observable[], frame, plot_mask)
            notify(color_observable)

            style = styles[render.var]
            cmap_observable[] = style[1]
            clim_observable[] = style[2]
            colorbar_label[] = VAR_LABELS[render.var]
            title_observable[] = "$(VAR_LABELS[render.var]) | model year $(round(render.ref.time / SECONDS_PER_YEAR, digits = 2)) | video year $(round(render.video_year, digits = 2)) / $(length(vars) * DISPLAY_YEARS_PER_VARIABLE)"
            dot_x[] = render.ref.time / SECONDS_PER_YEAR
            dot_y[] = interpolate_series_value(ohc_series.time_years, ohc_series.ohc_anomaly, dot_x[])

            elevation_deg, azimuth_deg = camera_angles_deg(render.video_year, total_video_years)
            ax.elevation = deg2rad(elevation_deg)
            ax.azimuth = deg2rad(azimuth_deg)

            if frame_index == 1 || frame_index == total_frames || frame_index % 50 == 0
                @info "Rendering Earth-video frame" frame = frame_index total_frames variable = render.var model_year = round(render.ref.time / SECONDS_PER_YEAR, digits = 2)
            end

            save(png_path, fig)

            if frame_index % GC_INTERVAL == 0
                GC.gc(false)
            end
        end
    finally
        foreach(close_source_file!, values(sources))
    end

    @info "Saved Earth-video PNG sequence" frames_dir total_frames
    return assemble_earth_video_from_pngs(frames_dir, total_frames; outname, framerate)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_earth_video()
end
