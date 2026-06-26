using NumericalEarth
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Glob
using Oceananigans.Fields: location, interior
using JLD2
using OceanEnsembles

output_path = expanduser("../../outputs/saved/")
figdir = expanduser("../../figures/")
frames_dir = joinpath(figdir, "animation_frames")

resolution = "sxtdeg"
preview_offset = 19
static_elevation_deg = 20.0
static_azimuth_deg = -35.0
seconds_per_iteration = 10 * 60
seconds_per_spin = 3 * 365 * 24 * 60 * 60
video_framerate = 12
video_outname = joinpath(figdir, "all_vars_earth_vid.mp4")

grid = jldopen(glob("combined_global_75_fields_$(resolution)_RYF_run*.jld2", output_path)[1])["serialized/grid"];

files_combined = filter(f -> !occursin("_rank", f),
                        glob("global_*$(resolution)_RYF_run*.jld2", output_path))

Tfield = CenterField(grid)
ufield = XFaceField(grid)
vfield = YFaceField(grid)

function spherical_coordinates_viz(λ, φ, r=1)
    λ_rad = deg2rad.(λ)
    φ_rad = deg2rad.(φ)

    x = @. r * cos(φ_rad) * cos(λ_rad)
    y = @. r * cos(φ_rad) * sin(λ_rad)
    z = @. r * sin(φ_rad)

    return x, y, z
end

Tℓx, Tℓy, Tℓz = location(Tfield)
Uℓx, Uℓy, Uℓz = location(ufield)
Vℓx, Vℓy, Vℓz = location(vfield)

λ = λnodes(grid.underlying_grid, Tℓx(), Tℓy(), Tℓz())
φ = φnodes(grid.underlying_grid, Tℓx(), Tℓy(), Tℓz())
Tx, Ty, Tz = spherical_coordinates_viz(λ, φ, 1)

λ = λnodes(grid.underlying_grid, Uℓx(), Uℓy(), Uℓz())
φ = φnodes(grid.underlying_grid, Uℓx(), Uℓy(), Uℓz())
Ux, Uy, Uz = spherical_coordinates_viz(λ, φ, 1)

λ = λnodes(grid.underlying_grid, Vℓx(), Vℓy(), Vℓz())
φ = φnodes(grid.underlying_grid, Vℓx(), Vℓy(), Vℓz())
Vx, Vy, Vz = spherical_coordinates_viz(λ, φ, 1)

earth_underlay_color = RGBf(0.80, 0.82, 0.78)

n = 1024 ÷ 4
lat = reverse(LinRange(-π / 2, π / 2, n))
lon = LinRange(-π, π, 2 * n)

r = 1.0
r_offset = -0.02

x = [(r + r_offset) * cos(lat) * cos(lon) for lat in lat, lon in lon]
y = [(r + r_offset) * cos(lat) * sin(lon) for lat in lat, lon in lon]
z = [(r + r_offset) * sin(lat) for lat in lat, lon in lon]

depth_levels = [parse(Int, match(r"global_(\d+)", f).captures[1])
                for f in files_combined if occursin(r"global_\d+", f)]
unique_depth_levels = sort(unique(depth_levels))
depths_actual = abs.(grid.z.cᵃᵃᶠ[unique_depth_levels])

iterations = [match(r"run(\d+)", f).captures[1]
              for f in files_combined if occursin(r"run\d+", f)]
unique_iterations = sort(unique(iterations))

mutable struct VariableFrameSource
    var::String
    filepaths::Vector{String}
    file_runs::Vector{String}
    frame_counts::Vector{Int}
    cumulative_frames::Vector{Int}
    frame_times::Vector{Float64}
    frame_iterations::Vector{Int}
    current_file_idx::Int
    file_handle::Any
    raw_series::Any
    u_series::Any
    v_series::Any
end

function close_source_file!(source::VariableFrameSource)
    if source.file_handle !== nothing
        close(source.file_handle)
        source.file_handle = nothing
    end

    source.raw_series = nothing
    source.u_series = nothing
    source.v_series = nothing
    source.current_file_idx = 0
    return nothing
end

function validate_timeseries_file(filepath::String, var::String, is_speed::Bool)
    try
        if is_speed
            FieldTimeSeries(filepath, "u")
            FieldTimeSeries(filepath, "v")
        else
            FieldTimeSeries(filepath, var)
        end
        return true
    catch err
        @warn "Skipping unreadable or incomplete timeseries file." filepath var exception=(err, catch_backtrace())
        return false
    end
end

function build_variable_source(var::String, iterations::Vector{SubString{String}}; depth::Int=75)
    is_speed = (var == "speed")
    filepaths = String[]
    file_runs = String[]
    frame_counts = Int[]
    frame_times = Float64[]
    frame_iterations = Int[]

    for iteration in iterations
        filepath = output_path * "global_$(depth)_fields_$(resolution)_RYF_run$(iteration).jld2"
        @info "Indexing $var at iteration $iteration"

        try
            jldopen(filepath, "r") do f
                var_exists = is_speed ? (haskey(f, "timeseries/u") && haskey(f, "timeseries/v")) :
                                        haskey(f, "timeseries/$var")

                if !var_exists
                    @warn "Variable $var not found in $filepath. Skipping file."
                    return
                end

                validate_timeseries_file(filepath, var, is_speed) || return

                ts_keys = sort!(parse.(Int, collect(keys(f["timeseries/t"]))))
                append!(frame_times, [Float64(f["timeseries/t/$(key)"]) for key in ts_keys])
                append!(frame_iterations, ts_keys)
                push!(filepaths, filepath)
                push!(file_runs, String(iteration))
                push!(frame_counts, length(ts_keys))
            end
        catch err
            @warn "Skipping file during Earth-video indexing because it could not be read cleanly." filepath var exception=(err, catch_backtrace())
        end
    end

    isempty(frame_times) && error("No frames found for variable $var at depth $depth.")

    return VariableFrameSource(var,
                               filepaths,
                               file_runs,
                               frame_counts,
                               cumsum(frame_counts),
                               frame_times,
                               frame_iterations,
                               0,
                               nothing,
                               nothing,
                               nothing,
                               nothing)
end

function ensure_source_file!(source::VariableFrameSource, file_idx::Int)
    source.current_file_idx == file_idx && return nothing
    close_source_file!(source)

    filepath = source.filepaths[file_idx]

    try
        source.file_handle = jldopen(filepath, "r")

        if source.var == "speed"
            source.u_series = FieldTimeSeries(filepath, "u")
            source.v_series = FieldTimeSeries(filepath, "v")
        else
            source.raw_series = FieldTimeSeries(filepath, source.var)
        end
    catch err
        close_source_file!(source)
        rethrow(err)
    end

    source.current_file_idx = file_idx
    return nothing
end

frame_count(source::VariableFrameSource) = source.cumulative_frames[end]
frame_time(source::VariableFrameSource, frame_idx::Int) = source.frame_times[frame_idx]

function frame_location(source::VariableFrameSource, frame_idx::Int)
    file_idx = searchsortedfirst(source.cumulative_frames, frame_idx)
    local_frame = file_idx == 1 ? frame_idx : frame_idx - source.cumulative_frames[file_idx - 1]
    run_id = source.file_runs[file_idx]
    iteration = source.frame_iterations[frame_idx]
    return (; file_idx, local_frame, run_id, iteration)
end

function load_frame(source::VariableFrameSource, frame_idx::Int)
    loc = frame_location(source, frame_idx)
    ensure_source_file!(source, loc.file_idx)

    if source.var == "speed"
        u = interior(source.u_series[loc.local_frame])[:, :, 1]
        v = interior(source.v_series[loc.local_frame])[:, :, 1]
        return Array{Float32}(@. sqrt(u * u + v * v))
    end

    return Array{Float32}(interior(source.raw_series[loc.local_frame])[:, :, 1])
end

function variable_style(var::String, source::VariableFrameSource)
    if var == "S"
        return (; clim=(34.8f0, 37f0), cmap=:blues, coords=(Tx, Ty, Tz))
    elseif var == "u"
        return (; clim=(-0.5f0, 0.5f0), cmap=:balance, coords=(Ux, Uy, Uz))
    elseif var == "v"
        return (; clim=(-0.5f0, 0.5f0), cmap=:balance, coords=(Vx, Vy, Vz))
    elseif var == "w"
        return (; clim=(-2f-5, 2f-5), cmap=:balance, coords=(Tx, Ty, Tz))
    elseif var == "speed"
        return (; clim=(0f0, 0.7f0), cmap=:speed, coords=(Tx, Ty, Tz))
    elseif var == "T"
        last_frame = load_frame(source, frame_count(source))
        return (; clim=(minimum(last_frame), maximum(last_frame)), cmap=:inferno, coords=(Tx, Ty, Tz))
    else
        last_frame = load_frame(source, frame_count(source))
        return (; clim=(minimum(last_frame), maximum(last_frame)), cmap=:viridis, coords=(Tx, Ty, Tz))
    end
end

function export_frame_plan(vars::Vector{String}, sources::Dict{String, VariableFrameSource}; start_offset::Int=30)
    plan = Vector{Tuple{String, Int}}()

    for var in vars
        source_n = frame_count(sources[var])
        source_n > 0 || continue
        first_frame = min(source_n, max(1, start_offset + 1))
        append!(plan, ((var, frame_idx) for frame_idx in first_frame:source_n))
    end

    isempty(plan) && error("No export frames available after applying start_offset=$(start_offset).")
    return plan
end

masked_surface(A, land) = ifelse.(land, NaN32, A)

function camera_angles_deg(iteration::Int)
    elapsed_seconds = iteration * seconds_per_iteration
    azimuth_deg = static_azimuth_deg + 360 * (elapsed_seconds / seconds_per_spin)
    azimuth_deg = mod(azimuth_deg + 180, 360) - 180
    return static_elevation_deg, azimuth_deg
end

function frame_png_path(frames_dir::AbstractString, var::String, run_id::AbstractString, iteration::Int)
    filename = string(var, "_run", run_id, "_iter", lpad(iteration, 10, '0'), ".png")
    return joinpath(frames_dir, filename)
end

function parse_frame_png(path::AbstractString)
    m = match(r"^([A-Za-z0-9_]+)_iter(\d+)\.png$", basename(path))
    m === nothing && return nothing
    return (var = m.captures[1], iteration = parse(Int, m.captures[2]), path = path)
end

function assemble_earth_video_from_pngs(frames_dir::AbstractString,
                                        vars::Vector{String},
                                        sources::Dict{String, VariableFrameSource},
                                        plan::Vector{Tuple{String, Int}};
                                        outname::String=video_outname,
                                        framerate::Int=video_framerate)
    ffmpeg = Sys.which("ffmpeg")
    if isnothing(ffmpeg)
        @warn "Skipping MP4 assembly because `ffmpeg` is not available on PATH." frames_dir outname
        return nothing
    end

    expected_pngs = String[]
    counts_by_var = Dict(var => 0 for var in vars)

    for (var, frame_idx) in plan
        loc = frame_location(sources[var], frame_idx)
        png_path = frame_png_path(frames_dir, var, loc.run_id, loc.iteration)
        isfile(png_path) || error("Missing PNG frame for Earth video assembly: " * png_path)
        push!(expected_pngs, png_path)
        counts_by_var[var] += 1
    end

    stage_dir = mktempdir(; prefix = "earth_video_frames_")

    try
        for (i, png_path) in enumerate(expected_pngs)
            staged = joinpath(stage_dir, "frame_" * lpad(i, 6, '0') * ".png")
            symlink(abspath(png_path), staged)
        end

        input_pattern = joinpath(stage_dir, "frame_%06d.png")
        cmd = `$ffmpeg -y -framerate $framerate -i $input_pattern -pix_fmt yuv420p $outname`
        @info "Assembling Earth video from cached PNGs..." frames = length(expected_pngs) counts_by_var outname ffmpeg
        run(cmd)
        @info "Saved Earth video." outname
    finally
        rm(stage_dir; recursive = true, force = true)
    end

    return outname
end

function export_concatenated_variable_frames(vars::Vector{String},
                                             depths::Vector{Int},
                                             depth_actual::Vector{Float64},
                                             iterations::Vector{SubString{String}};
                                             frames_dir::String=frames_dir,
                                             start_offset::Int=30)
    depth = 75
    mkpath(frames_dir)

    sources = Dict{String, VariableFrameSource}()
    styles = Dict{String, NamedTuple}()

    for var in vars
        @info "Preparing $var for Earth snapshots..."
        source = build_variable_source(var, iterations; depth)
        sources[var] = source
        styles[var] = variable_style(var, source)
    end

    plan = export_frame_plan(vars, sources; start_offset)
    total_frames = length(plan)

    first_var, first_local_frame = first(plan)
    style0 = styles[first_var]
    cx0, cy0, cz0 = style0.coords

    land = view((grid.immersed_boundary.bottom_height .== 0), :, :, 1)

    X = Observable(cx0)
    Y = Observable(cy0)
    Zgeo = Observable(cz0)
    C = Observable(masked_surface(load_frame(sources[first_var], first_local_frame), land))
    cmap = Observable(style0.cmap)
    clim = Observable(style0.clim)

    fig = Figure(size=(750, 800))
    gl = fig[1, 1] = GridLayout()
    ax = Axis3(gl[2, 1], aspect=:data, viewmode=:fitzoom)

    surface!(ax, x, y, z;
             color=earth_underlay_color,
             shading=NoShading,
             backlight=1.5f0)

    hm = surface!(ax, X, Y, Zgeo;
                  color=C,
                  colormap=cmap,
                  colorrange=clim,
                  nan_color=earth_underlay_color)

    hidedecorations!(ax)
    hidespines!(ax)

    fig_title = Label(gl[1, 1], "Loading...", tellwidth=false)
    cb = Colorbar(gl[3, 1], hm, label=first_var, vertical=false)

    current_var = Ref("")

    try
        for (order_idx, (var, frame_idx)) in enumerate(plan)
            loc = frame_location(sources[var], frame_idx)
            png_path = frame_png_path(frames_dir, var, loc.run_id, loc.iteration)

            if isfile(png_path)
                @info "Skipping existing Earth snapshot." frame = order_idx variable = var path = png_path
                continue
            end

            style = styles[var]
            if current_var[] != var
                current_var[] = var
                cx, cy, cz = style.coords
                X[] = cx
                Y[] = cy
                Zgeo[] = cz
                cmap[] = style.cmap
                clim[] = style.clim
                cb.label[] = var
                GC.gc(false)
            end

            year = frame_time(sources[var], frame_idx) / (365 * 24 * 60 * 60)
            fig_title.text[] = "Var: $var - Year = $(round(year, digits=2)) - Iteration $(loc.iteration)"
            C[] = masked_surface(load_frame(sources[var], frame_idx), land)
            elevation_deg, azimuth_deg = camera_angles_deg(loc.iteration)
            ax.elevation = deg2rad(elevation_deg)
            ax.azimuth = deg2rad(azimuth_deg)

            save(png_path, fig)
            @info "Saved Earth snapshot." frame = order_idx total_frames variable = var iteration = loc.iteration sampled_frame = loc.local_frame path = png_path
        end
    finally
        for source in values(sources)
            close_source_file!(source)
        end
    end

    @info "Earth snapshot export complete." frames_dir total_frames
    assemble_earth_video_from_pngs(frames_dir, vars, sources, plan)
    return nothing
end

####################################################################
# RUN
####################################################################

vars = ["T", "S", "speed", "w"]
export_concatenated_variable_frames(vars, unique_depth_levels, depths_actual, unique_iterations;
                                    frames_dir=frames_dir,
                                    start_offset=preview_offset)
