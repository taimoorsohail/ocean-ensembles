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
const DEPTH_FILE_RUN_PREFIX = "*75_fields_$(RESOLUTION)_RYF_*"

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
    files = glob("*75_fields_$(RESOLUTION)_RYF_*.jld2", path)
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

function surface_frame_records(files::Vector{String}, vars::Vector{String})
    records_by_time = Dict{Float64, NamedTuple}()

    if file_has_timeseries_layout(first(files))
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
                    existing = get(records_by_time, tval, nothing)
                    if isnothing(existing) || run >= existing.run
                        records_by_time[tval] = (layout = :timeseries, run = run, file = file, key = key)
                    end
                end
            end
        end
    else
        for (file_index, file) in enumerate(files)
            jldopen(file, "r") do f
                haskey(f, "time") || return
                all(haskey(f, var) for var in vars) || return

                tval = Float64(f["time"])
                existing = get(records_by_time, tval, nothing)
                if isnothing(existing) || file_index >= existing.file_index
                    records_by_time[tval] = (layout = :top_level, file_index = file_index, file = file)
                end
            end
        end
    end

    times = sort(collect(keys(records_by_time)))
    records = [records_by_time[t] for t in times]

    return times, records
end

function load_surface_frame(record, var::String)
    return jldopen(record.file, "r") do f
        raw = if record.layout === :timeseries
            f["timeseries/$var/$(record.key)"]
        else
            f[var]
        end

        A = extract_2d_f32(raw)
        A === nothing && error("Unable to extract 2D Float32 field for `$var` from $(record.file).")
        return A
    end
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

function face_lon_lat(grid)
    g = hasproperty(grid, :underlying_grid) ? grid.underlying_grid : grid

    λ = λnodes(g, Face(), Face(), Center())
    φ = φnodes(g, Face(), Face(), Center())

    λA = Array(λ)
    φA = Array(φ)

    ndims(λA) == 3 && (λA = λA[:, :, 1])
    ndims(φA) == 3 && (φA = φA[:, :, 1])

    return Float32.(λA), Float32.(φA)
end

function extend_surface_field(field::AbstractMatrix{<:Real}, mask::BitMatrix)
    Nx, Ny = size(field)
    size(mask) == (Nx, Ny) || error("Mask shape mismatch: got $(size(mask)) expected $(size(field)).")

    extended = fill(NaN32, Nx + 1, Ny + 1)

    @inbounds for j in 1:Ny, i in 1:Nx
        mask[i, j] || continue
        extended[i, j] = Float32(field[i, j])
    end

    @inbounds for i in 1:Nx
        if mask[i, Ny]
            extended[i, Ny + 1] = Float32(field[i, Ny])
        end
    end

    @inbounds for j in 1:Ny + 1
        extended[Nx + 1, j] = extended[1, j]
    end

    return extended
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

function make_antarctic_surface_w_video(files::Vector{String}, grid;
                                        outname::Union{Nothing, String} = nothing,
                                        framerate::Int = VIDEO_FRAMERATE,
                                        latitude_cutoff::Real = 55)
    times, records = surface_frame_records(files, ["w"])
    nframes = length(times)
    nframes > 0 || error("No frames available for Antarctic surface w animation.")

    lon, lat = center_lon_lat(grid)
    face_lon, face_lat = face_lon_lat(grid)
    cutoff = Float32(latitude_cutoff)
    mask = lat .<= -cutoff
    any(mask) || error("No points found south of -$(latitude_cutoff) degrees.")

    w0 = load_surface_frame(records[1], "w")
    size(w0) == size(lon) || error("Surface w shape mismatch: got $(size(w0)) expected $(size(lon)).")

    source_proj = "+proj=longlat +datum=WGS84"
    dest_proj = stereographic_projection(:south)
    Z = Observable(ifelse.(mask, w0, NaN32))
    Z_surface = Observable(extend_surface_field(w0, mask))
    years = times ./ SECONDS_PER_YEAR
    cmap = :balance
    clim = (-2f-5, 2f-5)

    fig = Figure(size = (1100, 950))
    fig_title = Label(fig[0, 1], "Antarctic loading...", tellwidth = false)
    ax = GeoAxis(fig[1, 1];
                 source = source_proj,
                 dest = dest_proj,
                 title = "Surface vertical velocity w (Antarctic)")
    hidedecorations!(ax)

    hm = try
        heatmap!(ax, lon, lat, Z; colormap = cmap, colorrange = clim)
    catch err
        @warn "GeoAxis heatmap path failed; falling back to surface plot with duplicated Ny row." exception = (err, catch_backtrace())
        surface!(ax,
                 face_lon,
                 face_lat,
                 zeros(Float32, size(face_lon));
                 color = Z_surface,
                 shading = NoShading,
                 colormap = cmap,
                 colorrange = clim)
    end
    Colorbar(fig[2, 1], hm, label = "Vertical Velocity (m/s)", vertical = false)

    xlims!(ax, -180, 180)
    ylims!(ax, -90, -latitude_cutoff)
    resize_to_layout!(fig)

    isnothing(outname) && (outname = FIGDIR * "w_$(RESOLUTION)_antarctic_surface.mp4")
    @info "Recording Antarctic surface w animation..." outname nframes framerate latitude_cutoff
    progress_step = max(1, cld(nframes, PROGRESS_UPDATES))
    record(fig, outname, 1:nframes; framerate = framerate) do frame
        w = load_surface_frame(records[frame], "w")
        fig_title.text = "Antarctic surface w | Year = $(round(years[frame], digits = 2))"
        Z[] = ifelse.(mask, w, NaN32)
        Z_surface[] = extend_surface_field(w, mask)
        if frame == 1 || frame == nframes || frame % progress_step == 0
            log_record_progress("antarctic_surface_w", frame, nframes)
        end
    end
    @info "Saved Antarctic surface w animation." outname
    return outname
end

function run_antarctic_animation()
    files = top_surface_files(OUTPUT_PATH)
    isempty(files) && error("No top-surface files found in $(OUTPUT_PATH).")

    grid = load_grid_from_output_file(first(files))
    make_antarctic_surface_w_video(files, grid)
    return nothing
end

run_antarctic_animation()
