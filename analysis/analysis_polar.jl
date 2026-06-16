using CairoMakie
using GeoMakie
using JLD2
using Glob
using Oceananigans
using Oceananigans.Fields: location
using NumericalEarth

with_trailing_slash(path) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator

const OUTPUT_PATH = with_trailing_slash(expanduser(get(ENV, "OUTPUT_PATH", "/data/gpfs/projects/punim2499/taimoor/ocean-ensembles/outputs/saved/")))
const FIGDIR = with_trailing_slash(expanduser(get(ENV, "FIGDIR", "/data/gpfs/projects/punim2499/taimoor/ocean-ensembles/figures/")))
const RESOLUTION = "sxtdeg"
const SECONDS_PER_YEAR = 365 * 24 * 60 * 60
const VIDEO_FRAMERATE = 12
const PROGRESS_UPDATES = 20
const NAN_PLOT_COLOR = :lightgray
const SEA_ICE_VARS = ["ice_concentration", "ice_thickness"]

pretty_var_name(var::String) = var == "ice_concentration" ? "Sea Ice Concentration" :
                               var == "ice_thickness" ? "Sea Ice Thickness" : replace(var, "_" => " ")

@inline function run_id(path::AbstractString)
    m = match(r"run(\d+)", basename(path))
    return m === nothing ? -1 : parse(Int, m.captures[1])
end

function parse_selected_run()
    isempty(ARGS) && return nothing
    any(arg -> arg in ("-h", "--help"), ARGS) && return :help
    length(ARGS) == 1 || throw(ArgumentError("Expected at most one positional argument: run number."))
    selected_run = tryparse(Int, ARGS[1])
    selected_run === nothing && throw(ArgumentError("Could not parse run number from \"$(ARGS[1])\"."))
    selected_run < 0 && throw(ArgumentError("Run number must be non-negative, got $(selected_run)."))
    return selected_run
end

function print_usage()
    println("Usage: julia --project=ocean-ensembles/experiments/submission_scripts ocean-ensembles/analysis/analysis_polar.jl [run_id]")
    println("Builds a sea-ice evolution animation from combined surface-flux files.")
    return nothing
end

run_suffix(run::Int) = "run" * lpad(string(run), 4, '0')

function sea_ice_surface_files(path::AbstractString)
    files = glob("combined_global_surface_fluxes_$(RESOLUTION)_RYF_run*.jld2", path)
    files = filter(files) do f
        !occursin("_rank", f) && run_id(f) >= 0
    end
    sort!(files; by = run_id)
    return files
end

function filter_files_by_run(files::Vector{String}, selected_run::Union{Nothing, Int})
    isnothing(selected_run) && return files
    return filter(f -> run_id(f) == selected_run, files)
end

@inline function raw_surface_slice(raw)
    if ndims(raw) == 2
        return raw
    elseif ndims(raw) == 3
        size(raw, 3) >= 1 || return nothing
        return @view raw[:, :, 1]
    end
    return nothing
end

function copy_surface_slice!(dest::AbstractMatrix{Float32}, raw)
    raw2d = raw_surface_slice(raw)
    raw2d === nothing && return false
    size(dest) == size(raw2d) || error("Surface field shape mismatch: got $(size(raw2d)) expected $(size(dest)).")
    @inbounds for i in eachindex(dest, raw2d)
        dest[i] = Float32(raw2d[i])
    end
    return true
end

function finite_maximum_in_raw(raw; default = 0f0, context::AbstractString = "field")
    raw2d = raw_surface_slice(raw)
    raw2d === nothing && return default

    maxval = -Inf32
    found_finite = false
    @inbounds for value in raw2d
        if isfinite(value)
            maxval = max(maxval, Float32(value))
            found_finite = true
        end
    end

    if !found_finite
        @warn "No finite values found for maximum; using default value." context default
        return default
    end

    return maxval
end

function scan_surface_timeseries_refs(files::Vector{String}, vars::Vector{String})
    records_by_time = Dict{Float64, NamedTuple{(:run, :file, :key), Tuple{Int, String, Int}}}()
    replaced_duplicates = 0
    @info "Scanning sea-ice surface timeseries metadata..." file_count = length(files) variables = vars

    stride = max(1, cld(length(files), PROGRESS_UPDATES))
    for (file_index, file) in enumerate(files)
        run = run_id(file)
        jldopen(file, "r") do f
            has_t = haskey(f, "timeseries/t")
            missing = [v for v in vars if !haskey(f, "timeseries/$v")]
            if !has_t || !isempty(missing)
                @warn "Skipping surface-timeseries file: required fields missing." file missing
                return
            end

            ts_keys = sort(parse.(Int, collect(keys(f["timeseries/t"]))))
            for key in ts_keys
                tval = Float64(f["timeseries/t/$key"])
                existing = get(records_by_time, tval, nothing)
                if isnothing(existing) || run >= existing.run
                    replaced_duplicates += !isnothing(existing) && run > existing.run ? 1 : 0
                    records_by_time[tval] = (run = run, file = file, key = key)
                end
            end
        end

        if file_index == 1 || file_index == length(files) || file_index % stride == 0
            pct = round(100 * file_index / length(files); digits = 1)
            @info "Sea-ice metadata scan" progress = "$(file_index)/$(length(files))" percent = pct
        end
    end

    all_time = sort(collect(keys(records_by_time)))
    frame_refs = [(time = t, file = records_by_time[t].file, key = records_by_time[t].key, run = records_by_time[t].run) for t in all_time]

    @info "Completed sea-ice metadata scan." frames = length(frame_refs) variables = vars replaced_duplicates
    return frame_refs
end

function compute_surface_maxima(frame_refs, vars::Vector{String})
    maxima = Dict{String, Float32}(var => -Inf32 for var in vars)
    isempty(frame_refs) && return maxima

    stride = max(1, cld(length(frame_refs), PROGRESS_UPDATES))
    @info "Computing sea-ice color limits..." frames = length(frame_refs) variables = vars
    for (frame_index, ref) in enumerate(frame_refs)
        jldopen(ref.file, "r") do f
            for var in vars
                maxima[var] = max(maxima[var], finite_maximum_in_raw(f["timeseries/$var/$(ref.key)"]; default = 0f0, context = var))
            end
        end

        if frame_index == 1 || frame_index == length(frame_refs) || frame_index % stride == 0
            pct = round(100 * frame_index / length(frame_refs); digits = 1)
            @info "Sea-ice maxima scan" progress = "$(frame_index)/$(length(frame_refs))" percent = pct
        end
    end

    return maxima
end

function load_frame_data!(dest_by_var::Dict{String, Matrix{Float32}}, frame_ref, vars::Vector{String})
    jldopen(frame_ref.file, "r") do f
        for var in vars
            copy_surface_slice!(dest_by_var[var], f["timeseries/$var/$(frame_ref.key)"]) ||
                error("Unsupported surface field dimensions for variable=$(var), file=$(frame_ref.file), key=$(frame_ref.key).")
        end
    end
    return dest_by_var
end

function load_grid_from_output_file(filepath::AbstractString)
    grid = jldopen(filepath, "r") do f
        haskey(f, "serialized/grid") ? f["serialized/grid"] : nothing
    end
    grid === nothing && error("No serialized grid found in $(filepath).")
    return grid
end

function bottom_height_matrix(filepath::AbstractString)
    return jldopen(filepath, "r") do f
        haskey(f, "serialized/grid") || return nothing
        grid = f["serialized/grid"]
        hasproperty(grid, :immersed_boundary) || return nothing
        immersed_boundary = getproperty(grid, :immersed_boundary)
        hasproperty(immersed_boundary, :bottom_height) || return nothing
        bottom_height_field = getproperty(immersed_boundary, :bottom_height)
        bottom_height = Array(interior(bottom_height_field, :, :, 1))
        Float32.(bottom_height)
    end
end

surface_ocean_mask(bottom_height::Union{Nothing, AbstractMatrix}) = isnothing(bottom_height) ? nothing : bottom_height .< 0f0

function apply_plot_mask!(dest::AbstractMatrix{Float32}, src::AbstractMatrix{Float32}, plot_mask::Union{Nothing, AbstractMatrix{Bool}})
    size(dest) == size(src) || error("Destination/source shape mismatch: got $(size(dest)) expected $(size(src)).")
    if isnothing(plot_mask)
        @inbounds for i in eachindex(dest, src)
            dest[i] = src[i]
        end
        return dest
    end

    size(dest) == size(plot_mask) || error("Plot mask shape mismatch: got $(size(plot_mask)) expected $(size(dest)).")
    @inbounds for i in eachindex(dest, src, plot_mask)
        dest[i] = plot_mask[i] ? src[i] : NaN32
    end
    return dest
end

function active_column_range(mask::AbstractMatrix{Bool})
    active_columns = vec(any(mask, dims = 1))
    first_col = findfirst(active_columns)
    last_col = findlast(active_columns)
    isnothing(first_col) && error("No active columns found in hemisphere mask.")
    isnothing(last_col) && error("No active columns found in hemisphere mask.")
    return first_col:last_col
end

function center_lon_lat(grid)
    cfield = CenterField(grid)
    ℓx, ℓy, ℓz = location(cfield)
    g = hasproperty(grid, :underlying_grid) ? grid.underlying_grid : grid

    λ = λnodes(g, ℓx(), ℓy(), ℓz())
    φ = φnodes(g, ℓx(), ℓy(), ℓz())

    λA = Array(λ)
    φA = Array(φ)

    ndims(λA) == 3 && (λA = @view λA[:, :, 1])
    ndims(φA) == 3 && (φA = @view φA[:, :, 1])

    return Float32.(λA), Float32.(φA)
end

function stereographic_projection(hemisphere::Symbol)
    if hemisphere === :north
        return "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=0 +datum=WGS84"
    elseif hemisphere === :south
        return "+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +datum=WGS84"
    else
        throw(ArgumentError("hemisphere must be :north or :south."))
    end
end

function sea_ice_color_settings(var::String, vmax::Float32)
    if var == "ice_concentration"
        return :ice, vmax <= 1.2f0 ? (0f0, 1f0) : (0f0, 100f0)
    elseif var == "ice_thickness"
        return :ice, (0f0, max(1f0, vmax))
    else
        error("Unsupported sea-ice variable $var")
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

function make_sea_ice_evolution(selected_run::Union{Nothing, Int} = parse_selected_run();
                                outname::Union{Nothing, String} = nothing,
                                framerate::Int = VIDEO_FRAMERATE,
                                latitude_cutoff::Real = 50)
    selected_run === :help && return print_usage()

    files = filter_files_by_run(sea_ice_surface_files(OUTPUT_PATH), selected_run)
    isempty(files) && error("No combined sea-ice source files found in $(OUTPUT_PATH).")

    frame_refs = scan_surface_timeseries_refs(files, SEA_ICE_VARS)
    nframes = length(frame_refs)
    nframes > 0 || error("No sea-ice frames found in combined surface-flux files.")

    grid = load_grid_from_output_file(first(files))
    bottom_height = bottom_height_matrix(first(files))
    lon, lat = center_lon_lat(grid)
    surface_mask = surface_ocean_mask(bottom_height)

    color_maxima = compute_surface_maxima(frame_refs, SEA_ICE_VARS)
    years = [frame_refs[i].time / SECONDS_PER_YEAR for i in 1:nframes]
    conc_cmap, conc_clim = sea_ice_color_settings("ice_concentration", get(color_maxima, "ice_concentration", 1f0))
    thick_cmap, thick_clim = sea_ice_color_settings("ice_thickness", get(color_maxima, "ice_thickness", 1f0))

    masks = Dict{Symbol, BitMatrix}()
    cutoff = Float32(latitude_cutoff)
    for hemisphere in (:north, :south)
        latitude_mask = hemisphere === :north ? lat .>= cutoff : lat .<= -cutoff
        any(latitude_mask) || error("No points found for hemisphere=$(hemisphere) with cutoff=$(latitude_cutoff).")
        masks[hemisphere] = BitMatrix(isnothing(surface_mask) ? latitude_mask : (latitude_mask .& surface_mask))
    end

    hemisphere_cols = Dict{Symbol, UnitRange{Int}}(
        :north => active_column_range(masks[:north]),
        :south => active_column_range(masks[:south])
    )
    hemisphere_lon = Dict{Symbol, Matrix{Float32}}()
    hemisphere_lat = Dict{Symbol, Matrix{Float32}}()
    hemisphere_masks = Dict{Symbol, BitMatrix}()
    hemisphere_surface_z = Dict{Symbol, Matrix{Float32}}()
    for hemisphere in (:north, :south)
        cols = hemisphere_cols[hemisphere]
        hemisphere_lon[hemisphere] = Matrix{Float32}(@view lon[:, cols])
        hemisphere_lat[hemisphere] = Matrix{Float32}(@view lat[:, cols])
        hemisphere_masks[hemisphere] = BitMatrix(@view masks[hemisphere][:, cols])
        hemisphere_surface_z[hemisphere] = zeros(Float32, size(hemisphere_lon[hemisphere]))
    end

    field_size = size(lon)
    frame_data = Dict{String, Matrix{Float32}}(var => Matrix{Float32}(undef, field_size...) for var in SEA_ICE_VARS)
    plot_buffers = Dict{Tuple{Symbol, String}, Matrix{Float32}}()
    for hemisphere in (:north, :south), var in SEA_ICE_VARS
        plot_buffers[(hemisphere, var)] = fill(NaN32, size(hemisphere_lon[hemisphere])...)
    end

    load_frame_data!(frame_data, first(frame_refs), SEA_ICE_VARS)
    for hemisphere in (:north, :south)
        cols = hemisphere_cols[hemisphere]
        apply_plot_mask!(plot_buffers[(hemisphere, "ice_concentration")], @view(frame_data["ice_concentration"][:, cols]), hemisphere_masks[hemisphere])
        apply_plot_mask!(plot_buffers[(hemisphere, "ice_thickness")], @view(frame_data["ice_thickness"][:, cols]), hemisphere_masks[hemisphere])
    end

    observables = Dict(
        (:north, "ice_concentration") => Observable(plot_buffers[(:north, "ice_concentration")]),
        (:south, "ice_concentration") => Observable(plot_buffers[(:south, "ice_concentration")]),
        (:north, "ice_thickness") => Observable(plot_buffers[(:north, "ice_thickness")]),
        (:south, "ice_thickness") => Observable(plot_buffers[(:south, "ice_thickness")])
    )

    fig = Figure(size = (1600, 1200))
    fig_title = Label(fig[0, :], "Sea-ice evolution loading...", tellwidth = false)
    source_proj = "+proj=longlat +datum=WGS84"

    panel_specs = [
        (row = 1, col = 1, hemisphere = :north, var = "ice_concentration", title = "Arctic Sea Ice Concentration", cmap = conc_cmap, clim = conc_clim),
        (row = 1, col = 2, hemisphere = :south, var = "ice_concentration", title = "Antarctic Sea Ice Concentration", cmap = conc_cmap, clim = conc_clim),
        (row = 2, col = 1, hemisphere = :north, var = "ice_thickness", title = "Arctic Sea Ice Thickness", cmap = thick_cmap, clim = thick_clim),
        (row = 2, col = 2, hemisphere = :south, var = "ice_thickness", title = "Antarctic Sea Ice Thickness", cmap = thick_cmap, clim = thick_clim),
    ]

    concentration_handle = nothing
    thickness_handle = nothing

    for spec in panel_specs
        ax = GeoAxis(fig[spec.row, spec.col];
                     source = source_proj,
                     dest = stereographic_projection(spec.hemisphere),
                     title = spec.title)
        hidedecorations!(ax)
        hm = surface!(ax,
                      hemisphere_lon[spec.hemisphere],
                      hemisphere_lat[spec.hemisphere],
                      hemisphere_surface_z[spec.hemisphere];
                      color = observables[(spec.hemisphere, spec.var)],
                      shading = NoShading,
                      colormap = spec.cmap,
                      colorrange = spec.clim,
                      nan_color = NAN_PLOT_COLOR)
        xlims!(ax, -180, 180)
        if spec.hemisphere === :north
            ylims!(ax, latitude_cutoff, 90)
        else
            ylims!(ax, -90, -latitude_cutoff)
        end
        if spec.var == "ice_concentration"
            concentration_handle = hm
        else
            thickness_handle = hm
        end
    end

    Colorbar(fig[3, :], concentration_handle, label = pretty_var_name("ice_concentration"), vertical = false)
    Colorbar(fig[4, :], thickness_handle, label = pretty_var_name("ice_thickness"), vertical = false)
    resize_to_layout!(fig)

    if isnothing(outname)
        outname = isnothing(selected_run) ? FIGDIR * "sea_ice_evolution.mp4" : FIGDIR * "sea_ice_evolution_$(run_suffix(selected_run)).mp4"
    end

    @info "Recording polar sea-ice animation..." outname nframes framerate latitude_cutoff
    progress_step = max(1, cld(nframes, PROGRESS_UPDATES))
    record(fig, outname, 1:nframes; framerate = framerate) do frame
        fig_title.text = "Sea-Ice Evolution | Year = $(round(years[frame], digits = 2))"
        load_frame_data!(frame_data, frame_refs[frame], SEA_ICE_VARS)
        for hemisphere in (:north, :south)
            cols = hemisphere_cols[hemisphere]
            apply_plot_mask!(plot_buffers[(hemisphere, "ice_concentration")], @view(frame_data["ice_concentration"][:, cols]), hemisphere_masks[hemisphere])
            apply_plot_mask!(plot_buffers[(hemisphere, "ice_thickness")], @view(frame_data["ice_thickness"][:, cols]), hemisphere_masks[hemisphere])
            notify(observables[(hemisphere, "ice_concentration")])
            notify(observables[(hemisphere, "ice_thickness")])
        end
        if frame == 1 || frame == nframes || frame % progress_step == 0
            log_record_progress("sea_ice_evolution", frame, nframes)
        end
    end

    @info "Saved polar sea-ice animation." outname
    return outname
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_sea_ice_evolution()
end
