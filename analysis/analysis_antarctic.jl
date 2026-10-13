using CairoMakie
using GeoMakie
using JLD2
using Glob
using Oceananigans
using Oceananigans.Fields: location
using Oceananigans.BoundaryConditions: fill_halo_regions!

with_trailing_slash(path) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator

const OUTPUT_PATH = with_trailing_slash(expanduser(get(ENV, "OUTPUT_PATH", "/home/tsohail/uom/ocean-ensembles/outputs/saved/")))
const FIGDIR = with_trailing_slash(expanduser(get(ENV, "FIGDIR", "/home/tsohail/uom/ocean-ensembles/figures/")))
const RESOLUTION = "sxtdeg"
const SECONDS_PER_YEAR = 365 * 24 * 60 * 60
const VIDEO_FRAMERATE = 12
const PROGRESS_UPDATES = 20
const DEPTH_FILE_RUN_PREFIX = "75_fields_$(RESOLUTION)_RYF_run"

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
    files = glob("*75_fields_$(RESOLUTION)_RYF_run*.jld2", path)
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

    sort!(files; by = run_id)
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

# The speed helpers below are copied from analysis2d_horizontal.jl so both scripts compute speed the same way.
const OFFSET_ARRAYS = Base.loaded_modules[Base.PkgId(Base.UUID("6fe1bfb0-de20-5000-8ca7-80f57d26f881"), "OffsetArrays")]

underlying_grid(grid) = hasproperty(grid, :underlying_grid) ? getproperty(grid, :underlying_grid) : grid

function materialize_offset_array(source)
    if hasproperty(source, :parent) && hasproperty(source, :offsets)
        return OFFSET_ARRAYS.OffsetArray(getproperty(source, :parent), getproperty(source, :offsets)...)
    end

    return source
end

# Rebuild the saved grid as a plain CPU grid so Oceananigans can do calculations on it.
function materialized_underlying_grid(grid)
    g = underlying_grid(grid)

    return OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(CPU(),
        getproperty(g, :Nx), getproperty(g, :Ny), getproperty(g, :Nz),
        getproperty(g, :Hx), getproperty(g, :Hy), getproperty(g, :Hz),
        getproperty(g, :Lz),
        materialize_offset_array(getproperty(g, :λᶜᶜᵃ)),
        materialize_offset_array(getproperty(g, :λᶠᶜᵃ)),
        materialize_offset_array(getproperty(g, :λᶜᶠᵃ)),
        materialize_offset_array(getproperty(g, :λᶠᶠᵃ)),
        materialize_offset_array(getproperty(g, :φᶜᶜᵃ)),
        materialize_offset_array(getproperty(g, :φᶠᶜᵃ)),
        materialize_offset_array(getproperty(g, :φᶜᶠᵃ)),
        materialize_offset_array(getproperty(g, :φᶠᶠᵃ)),
        getproperty(g, :z),
        materialize_offset_array(getproperty(g, :Δxᶜᶜᵃ)),
        materialize_offset_array(getproperty(g, :Δxᶠᶜᵃ)),
        materialize_offset_array(getproperty(g, :Δxᶜᶠᵃ)),
        materialize_offset_array(getproperty(g, :Δxᶠᶠᵃ)),
        materialize_offset_array(getproperty(g, :Δyᶜᶜᵃ)),
        materialize_offset_array(getproperty(g, :Δyᶠᶜᵃ)),
        materialize_offset_array(getproperty(g, :Δyᶜᶠᵃ)),
        materialize_offset_array(getproperty(g, :Δyᶠᶠᵃ)),
        materialize_offset_array(getproperty(g, :Azᶜᶜᵃ)),
        materialize_offset_array(getproperty(g, :Azᶠᶜᵃ)),
        materialize_offset_array(getproperty(g, :Azᶜᶠᵃ)),
        materialize_offset_array(getproperty(g, :Azᶠᶠᵃ)),
        getproperty(g, :radius),
        getproperty(g, :conformal_mapping))
end

@inline surface_matrix_3d(A::AbstractMatrix) = reshape(A, size(A, 1), size(A, 2), 1)

# u and v live on the cell faces. Oceananigans moves them to the cell centres for us
# with `@at`, then we take sqrt(u² + v²) to get the speed.
# Building the grid and fields is slow, so we build them once here and reuse them for every frame.
function speed_workspace(grid)
    speed_grid = materialized_underlying_grid(grid)
    ufield = XFaceField(speed_grid)
    vfield = YFaceField(speed_grid)
    speed_field = @at (Center, Center, Nothing) sqrt(ufield^2 + vfield^2) |> Field
    return (; ufield, vfield, speed_field)
end

# Put one frame of u and v into the workspace and return the speed at the cell centres.
function surface_speed_at_centers(u::AbstractMatrix, v::AbstractMatrix, workspace)
    set!(workspace.ufield, surface_matrix_3d(u))
    set!(workspace.vfield, surface_matrix_3d(v))
    fill_halo_regions!(workspace.ufield)
    fill_halo_regions!(workspace.vfield)
    compute!(workspace.speed_field)
    return Float32.(Array(interior(workspace.speed_field)[:, :, 1]))
end

function load_surface_speed_frame(record, workspace)
    u = load_surface_frame(record, "u")
    v = load_surface_frame(record, "v")
    return surface_speed_at_centers(u, v, workspace)
end

# The helpers below read the sea-floor height from the saved grid. They are copied from
# analysis2d_horizontal.jl. The saved grid stores its arrays with extra "halo" cells around
# the edges, so we cut those off to get one value per ocean cell.
function parent_array(source)
    return hasproperty(source, :parent) ? getproperty(source, :parent) : Array(source)
end

function interior_start(source, dim::Int, fallback_halo::Int, interior_size::Int, stored_size::Int)
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
    error("Could not crop stored dimension " * string(stored_size) * " to interior size " * string(interior_size) * ".")
end

function physical_matrix(source, grid; T = Float64)
    Nx, Ny = getproperty(grid, :Nx), getproperty(grid, :Ny)
    Hx, Hy = getproperty(grid, :Hx), getproperty(grid, :Hy)
    data = parent_array(source)
    i0 = interior_start(source, 1, Hx, Nx, size(data, 1))
    j0 = interior_start(source, 2, Hy, Ny, size(data, 2))

    if ndims(data) == 2
        return T.(view(data, i0:i0+Nx-1, j0:j0+Ny-1))
    elseif ndims(data) == 3
        return T.(view(data, i0:i0+Nx-1, j0:j0+Ny-1, 1))
    end

    error("Expected a 2D or 3D stored grid array, got " * string(ndims(data)) * " dimensions.")
end

function bottom_height_matrix(grid)
    hasproperty(grid, :immersed_boundary) || return nothing

    immersed_boundary = getproperty(grid, :immersed_boundary)
    hasproperty(immersed_boundary, :bottom_height) || return nothing

    bottom_height_field = getproperty(immersed_boundary, :bottom_height)
    hasproperty(bottom_height_field, :data) || return nothing
    return physical_matrix(getproperty(bottom_height_field, :data), underlying_grid(grid); T = Float32)
end

# A cell is land when the sea floor is at or above sea level (height >= 0).
# If the grid has no sea-floor information, we treat cells with exactly zero speed as land.
function land_mask(grid, speed::AbstractMatrix)
    bottom_height = bottom_height_matrix(grid)
    if isnothing(bottom_height) || size(bottom_height) != size(speed)
        @warn "No usable sea-floor height in the grid; treating zero-speed cells as land."
        return speed .== 0
    end
    return bottom_height .>= 0f0
end

# Build the coastline as short line pieces along every cell edge that has land on one side
# and ocean on the other. Using the model's own cells means the line matches the grey land exactly.
# `face_lon` and `face_lat` hold the cell corners.
# `shift = (di, dj)` moves every corner index by di in longitude and dj in latitude, so that the
# line lines up with the coloured cells on the plot. With shift = (-1, -1) the corner of cell (i, j)
# is read from index (i - 1, j - 1). Use (0, 0) for no shift or (1, 1) to shift the other way.
function coastline_segments(land::AbstractMatrix{Bool}, region::AbstractMatrix{Bool}, face_lon, face_lat;
                            shift::Tuple{Int, Int} = (-1, -1))
    Nx, Ny = size(land)
    Nfx, Nfy = size(face_lon)
    di, dj = shift
    corner(i, j) = Point2f(face_lon[mod1(i + di, Nfx), j + dj], face_lat[mod1(i + di, Nfx), j + dj])
    segments = Point2f[]

    @inbounds for j in 1:Ny, i in 1:Nx
        region[i, j] || continue
        # Skip cells whose shifted corners would fall off the top or bottom of the grid.
        (1 <= j + dj && j + 1 + dj <= Nfy) || continue

        # Edge shared with the cell to the east (the grid wraps around in longitude).
        if land[i, j] != land[mod1(i + 1, Nx), j]
            push!(segments, corner(i + 1, j), corner(i + 1, j + 1))
        end

        # Edge shared with the cell to the north.
        if j < Ny && land[i, j] != land[i, j + 1]
            push!(segments, corner(i, j + 1), corner(i + 1, j + 1))
        end
    end

    return segments
end

# Copy the first longitude column onto the end. The grid wraps around the globe, and without
# this copy the plot leaves a thin empty strip between the last column and the first one.
close_longitude_seam(A::AbstractMatrix) = vcat(A, A[1:1, :])

# Draw a field on the map. Points where `mask` is false are left see-through.
function plot_masked_field!(ax, lon, lat, face_lon, face_lat, field, mask; colormap, colorrange)
    return try
        heatmap!(ax,
                 close_longitude_seam(lon),
                 close_longitude_seam(lat),
                 close_longitude_seam(ifelse.(mask, field, NaN32));
                 colormap, colorrange)
    catch err
        @warn "GeoAxis heatmap path failed; falling back to surface plot with duplicated Ny row." exception = (err, catch_backtrace())
        surface!(ax,
                 face_lon,
                 face_lat,
                 zeros(Float32, size(face_lon));
                 color = extend_surface_field(field, mask),
                 shading = NoShading,
                 colormap,
                 colorrange)
    end
end

# Plot a single snapshot of surface current speed south of `latitude_cutoff` degrees south,
# with grey land and a black coastline.
# By default this uses the last saved time; pass `frame` to choose a different one.
function make_antarctic_surface_speed_snapshot(files::Vector{String}, grid;
                                               outname::Union{Nothing, String} = nothing,
                                               frame::Union{Nothing, Int} = nothing,
                                               latitude_cutoff::Real = 40,
                                               clim = (0f0, 0.5f0),
                                               colormap = :inferno)
    times, records = surface_frame_records(files, ["u", "v"])
    nframes = length(times)
    nframes > 0 || error("No frames available for Antarctic surface speed snapshot.")
    isnothing(frame) && (frame = nframes)

    lon, lat = center_lon_lat(grid)
    face_lon, face_lat = face_lon_lat(grid)
    region = lat .<= -Float32(latitude_cutoff)
    any(region) || error("No points found south of -$(latitude_cutoff) degrees.")

    speed = load_surface_speed_frame(records[frame], speed_workspace(grid))
    size(speed) == size(lon) || error("Surface speed shape mismatch: got $(size(speed)) expected $(size(lon)).")

    land = land_mask(grid, speed)
    ocean_region = region .& .!land
    land_region = region .& land

    year = round(times[frame] / SECONDS_PER_YEAR, digits = 2)

    fig = Figure(size = (1300, 1150), fontsize = 52)
    ax = GeoAxis(fig[1, 1];
                 source = "+proj=longlat +datum=WGS84",
                 dest = stereographic_projection(:south))
    hidedecorations!(ax)

    # The model grid stops short of the South Pole, which leaves an empty disk in the middle.
    # Fill it with the land colour. It is drawn first, so the model cells sit on top of it.
    grid_south_edge = maximum(@view face_lat[:, 1]) + 0.5f0
    pole_disk = [Point2f(λ, grid_south_edge) for λ in -180:1:180]
    poly!(ax, pole_disk; color = :lightgray, strokewidth = 0)

    hm = plot_masked_field!(ax, lon, lat, face_lon, face_lat, speed, ocean_region;
                            colormap, colorrange = clim)
    plot_masked_field!(ax, lon, lat, face_lon, face_lat, ones(Float32, size(speed)), land_region;
                       colormap = [:lightgray, :lightgray], colorrange = (0f0, 2f0))
    linesegments!(ax, coastline_segments(land, region, face_lon, face_lat);
                  color = :black, linewidth = 1)

    # Label Antarctica at the centre of the circle, which is the South Pole.
    # It uses the same font size as the colorbar label.
    colorbar_labelsize = 56
    text!(ax, Point2f(0, -90);
          text = "Antarctica",
          fontsize = colorbar_labelsize,
          color = :black,
          align = (:center, :center),
          # Nudge the label to the right (in pixels) so it sits in the middle of the land.
          offset = (50, 0))

    # Vertical colorbar on the right. Making the map column square means the circle fills
    # the whole row height. The colorbar is then set to 80% of the row, so 80% of the circle's height.
    Colorbar(fig[1, 2], hm, label = "Current Speed (m/s)", vertical = true,
             labelsize = colorbar_labelsize, ticklabelsize = 48, width = 50,
             height = Relative(0.8))
    colsize!(fig.layout, 1, Aspect(1, 1.0))

    xlims!(ax, -180, 180)
    ylims!(ax, -90, -latitude_cutoff)
    resize_to_layout!(fig)

    isnothing(outname) && (outname = FIGDIR * "speed_$(RESOLUTION)_antarctic_surface.png")
    save(outname, fig)
    @info "Saved Antarctic surface speed snapshot." outname frame year latitude_cutoff
    return outname
end

# Make a movie of surface current speed south of `latitude_cutoff` degrees south, one frame per saved time.
function make_antarctic_surface_speed_video(files::Vector{String}, grid;
                                            outname::Union{Nothing, String} = nothing,
                                            framerate::Int = VIDEO_FRAMERATE,
                                            latitude_cutoff::Real = 50,
                                            clim = (0f0, 0.5f0))
    times, records = surface_frame_records(files, ["u", "v"])
    nframes = length(times)
    nframes > 0 || error("No frames available for Antarctic surface speed animation.")

    lon, lat = center_lon_lat(grid)
    face_lon, face_lat = face_lon_lat(grid)
    mask = lat .<= -Float32(latitude_cutoff)
    any(mask) || error("No points found south of -$(latitude_cutoff) degrees.")

    workspace = speed_workspace(grid)
    speed0 = load_surface_speed_frame(records[1], workspace)
    size(speed0) == size(lon) || error("Surface speed shape mismatch: got $(size(speed0)) expected $(size(lon)).")

    Z = Observable(ifelse.(mask, speed0, NaN32))
    Z_surface = Observable(extend_surface_field(speed0, mask))
    years = times ./ SECONDS_PER_YEAR
    cmap = :speed

    fig = Figure(size = (1100, 950))
    fig_title = Label(fig[0, 1], "Antarctic loading...", tellwidth = false)
    ax = GeoAxis(fig[1, 1];
                 source = "+proj=longlat +datum=WGS84",
                 dest = stereographic_projection(:south),
                 title = "Surface current speed (Antarctic)")
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
    Colorbar(fig[2, 1], hm, label = "Current Speed (m/s)", vertical = false)

    xlims!(ax, -180, 180)
    ylims!(ax, -90, -latitude_cutoff)
    resize_to_layout!(fig)

    isnothing(outname) && (outname = FIGDIR * "speed_$(RESOLUTION)_antarctic_surface.mp4")
    @info "Recording Antarctic surface speed animation..." outname nframes framerate latitude_cutoff
    progress_step = max(1, cld(nframes, PROGRESS_UPDATES))
    record(fig, outname, 1:nframes; framerate = framerate) do frame
        speed = load_surface_speed_frame(records[frame], workspace)
        fig_title.text = "Antarctic surface speed | Year = $(round(years[frame], digits = 2))"
        Z[] = ifelse.(mask, speed, NaN32)
        Z_surface[] = extend_surface_field(speed, mask)
        if frame == 1 || frame == nframes || frame % progress_step == 0
            log_record_progress("antarctic_surface_speed", frame, nframes)
        end
    end
    @info "Saved Antarctic surface speed animation." outname
    return outname
end

function run_antarctic_animation()
    files = top_surface_files(OUTPUT_PATH)
    isempty(files) && error("No top-surface files found in $(OUTPUT_PATH).")

    grid = load_grid_from_output_file(first(files))
    make_antarctic_surface_speed_snapshot(files, grid)
    return nothing
end

run_antarctic_animation()
