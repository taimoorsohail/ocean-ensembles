using CairoMakie
using GeoMakie
using JLD2
using Glob
using Oceananigans
using Oceananigans.Fields: location

with_trailing_slash(path) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator

const OUTPUT_PATH = with_trailing_slash(expanduser(get(ENV, "OUTPUT_PATH", "/home/tsohail/uom/ocean-ensembles/outputs/")))
const FIGDIR = with_trailing_slash(expanduser(get(ENV, "FIGDIR", "/home/tsohail/uom/ocean-ensembles/figures/")))
const RESOLUTION = "sxtdeg"
const SECONDS_PER_YEAR = 365 * 24 * 60 * 60
const VIDEO_FRAMERATE = 12
const PROGRESS_UPDATES = 20
const DEPTH_FILE_RUN_PREFIX = "*75_fields_$(RESOLUTION)_RYF_run0001*"

@inline function run_id(path::AbstractString)
    m = match(r"run(\d+)", basename(path))
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

function top_surface_files(path::AbstractString)
    files = glob("*75_fields_$(RESOLUTION)_RYF_run0001*.jld2", path)
    files = filter(files) do f
        !occursin("_rank", f) &&
        !occursin("sea_ice_surface", basename(f)) &&
        !occursin("surface_fluxes", basename(f))
    end

    if isempty(files)
        files = glob("*$(DEPTH_FILE_RUN_PREFIX)*.jld2", path)
        files = filter(files) do f
            !occursin("_rank", f)
        end
    end

    sort!(files)
    return files
end

function file_has_timeseries_layout(filepath::AbstractString)
    return jldopen(filepath, "r") do f
        haskey(f, "timeseries/t")
    end
end

function load_grid_from_output_file(filepath::AbstractString)
    grid = jldopen(filepath, "r") do f
        haskey(f, "serialized/grid") ? f["serialized/grid"] : nothing
    end

    if grid === nothing
        if file_has_timeseries_layout(filepath)
            prefix = replace(filepath, r"\.jld2$" => "")
            @info "Falling back to create_grid for legacy layout." filepath
            return create_grid(prefix; gridtype = "TripolarGrid")
        else
            error("No serialized grid found in $(filepath), and legacy grid reconstruction is unavailable for top-level files.")
        end
    end

    return grid
end

function load_top_level_timeseries(files::Vector{String}, vars::Vector{String})
    records_by_time = Dict{Float64, NamedTuple{(:file_index, :fields), Tuple{Int, Dict{String, Matrix{Float32}}}}}()

    for (file_index, file) in enumerate(files)
        jldopen(file, "r") do f
            haskey(f, "time") || return
            tval = Float64(f["time"])
            timestep_fields = Dict{String, Matrix{Float32}}()

            for var in vars
                haskey(f, var) || return
                A = extract_2d_f32(f[var])
                A === nothing && return
                timestep_fields[var] = A
            end

            existing = get(records_by_time, tval, nothing)
            if isnothing(existing) || file_index >= existing.file_index
                records_by_time[tval] = (file_index = file_index, fields = timestep_fields)
            end
        end
    end

    all_time = sort(collect(keys(records_by_time)))
    all_data = Dict{String, Vector{Matrix{Float32}}}(v => Matrix{Float32}[] for v in vars)
    for var in vars
        all_data[var] = [records_by_time[t].fields[var] for t in all_time]
    end

    return all_time, all_data
end

function load_surface_timeseries(files::Vector{String}, vars::Vector{String})
    records_by_time = Dict{Float64, NamedTuple{(:run, :fields), Tuple{Int, Vector{Matrix{Float32}}}}}()

    for file in files
        run = run_id(file)
        jldopen(file, "r") do f
            has_t = haskey(f, "timeseries/t")
            missing = [v for v in vars if !haskey(f, "timeseries/$v")]
            if !has_t || !isempty(missing)
                return
            end

            ts_keys = sort(parse.(Int, collect(keys(f["timeseries/t"]))))
            for key in ts_keys
                tval = Float64(f["timeseries/t/$key"])
                timestep_fields = Matrix{Float32}[]
                valid = true

                for var in vars
                    A = extract_2d_f32(f["timeseries/$var/$key"])
                    if A === nothing
                        valid = false
                        break
                    end
                    push!(timestep_fields, A)
                end

                valid || continue
                existing = get(records_by_time, tval, nothing)
                if isnothing(existing) || run >= existing.run
                    records_by_time[tval] = (run = run, fields = timestep_fields)
                end
            end
        end
    end

    all_time = sort(collect(keys(records_by_time)))
    all_data = Dict{String, Vector{Matrix{Float32}}}(v => Matrix{Float32}[] for v in vars)
    for (var_idx, var) in enumerate(vars)
        all_data[var] = [records_by_time[t].fields[var_idx] for t in all_time]
    end

    return all_time, all_data
end

function load_general_surface_timeseries(files::Vector{String}, vars::Vector{String})
    isempty(files) && error("No files were provided.")
    return file_has_timeseries_layout(first(files)) ? load_surface_timeseries(files, vars) : load_top_level_timeseries(files, vars)
end

function log_record_progress(tag::String, frame::Int, nframes::Int)
    width = 24
    fraction = frame / nframes
    filled = clamp(floor(Int, width * fraction), 0, width)
    bar = "[" * repeat("=", filled) * repeat(".", width - filled) * "]"
    percent = round(100 * fraction; digits = 1)
    @info "Recording progress." variable = tag frame nframes percent bar
end

function center_lon_lat(grid)
    cfield = CenterField(grid)
    ℓx, ℓy, ℓz = location(cfield)
    g = hasproperty(grid, :underlying_grid) ? grid.underlying_grid : grid

    λ = λnodes(g, ℓx(), ℓy(), ℓz())
    φ = φnodes(g, ℓx(), ℓy(), ℓz())

    λA = Array(λ)
    φA = Array(φ)

    ndims(λA) == 3 && (λA = λA[:, :, 1])
    ndims(φA) == 3 && (φA = φA[:, :, 1])

    return Float32.(λA), Float32.(φA)
end

function stereographic_projection(hemisphere::Symbol)
    if hemisphere === :north
        return "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=0 +datum=WGS84"
    elseif hemisphere === :south
        return "+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +datum=WGS84"
    else
        throw(ArgumentError("`hemisphere` must be :north or :south."))
    end
end

function make_arctic_surface_w_video(files::Vector{String}, grid;
                                     outname::Union{Nothing, String} = nothing,
                                     framerate::Int = VIDEO_FRAMERATE,
                                     latitude_cutoff::Real = 55)
    times, all_data = load_general_surface_timeseries(files, ["w"])
    nframes = length(times)
    nframes > 0 || error("No frames available for Arctic surface w animation.")

    lon, lat = center_lon_lat(grid)
    cutoff = Float32(latitude_cutoff)
    mask = lat .>= cutoff
    any(mask) || error("No points found north of $(latitude_cutoff) degrees.")

    w0 = all_data["w"][1]
    size(w0) == size(lon) || error("Surface w shape mismatch: got $(size(w0)) expected $(size(lon)).")

    source_proj = "+proj=longlat +datum=WGS84"
    dest_proj = stereographic_projection(:north)
    masked_frames = [ifelse.(mask, all_data["w"][i], NaN32) for i in 1:nframes]
    Z = Observable(masked_frames[1])
    years = times ./ SECONDS_PER_YEAR
    cmap = :balance
    clim = (-2f-5, 2f-5)

    fig = Figure(size = (1100, 950))
    fig_title = Label(fig[0, 1], "Arctic loading...", tellwidth = false)
    ax = GeoAxis(fig[1, 1];
                 source = source_proj,
                 dest = dest_proj,
                 title = "Surface vertical velocity w (Arctic)")
    hidedecorations!(ax)

    hm = try
        heatmap!(ax, lon, lat, Z; colormap = cmap, colorrange = clim)
    catch err
        @warn "GeoAxis heatmap path failed; falling back to surface plot." exception = (err, catch_backtrace())
        surface!(ax,
                 lon,
                 lat,
                 zeros(Float32, size(lon));
                 color = Z,
                 shading = NoShading,
                 colormap = cmap,
                 colorrange = clim)
    end
    Colorbar(fig[2, 1], hm, label = "Vertical Velocity (m/s)", vertical = false)

    xlims!(ax, -180, 180)
    ylims!(ax, latitude_cutoff, 90)
    resize_to_layout!(fig)

    isnothing(outname) && (outname = FIGDIR * "w_$(RESOLUTION)_arctic_surface.mp4")

    @info "Recording Arctic surface w animation..." outname nframes framerate latitude_cutoff
    progress_step = max(1, cld(nframes, PROGRESS_UPDATES))
    record(fig, outname, 1:nframes; framerate = framerate) do frame
        fig_title.text = "Arctic surface w | Year = $(round(years[frame], digits = 2))"
        Z[] = masked_frames[frame]
        if frame == 1 || frame == nframes || frame % progress_step == 0
            log_record_progress("arctic_surface_w", frame, nframes)
        end
    end

    @info "Saved Arctic surface w animation." outname
    return outname
end

function run_arctic_animation()
    files = top_surface_files(OUTPUT_PATH)
    isempty(files) && error("No top-surface files found in $(OUTPUT_PATH).")

    grid = load_grid_from_output_file(first(files))
    make_arctic_surface_w_video(files, grid)
    return nothing
end

run_arctic_animation()
