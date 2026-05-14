using CairoMakie
using JLD2
using Glob
using OceanEnsembles
using Oceananigans
using Oceananigans.Fields: location

with_trailing_slash(path) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator

const OUTPUT_PATH = with_trailing_slash(expanduser(get(ENV, "OUTPUT_PATH", "/home/tsohail/uom/ocean-ensembles/outputs/")))
const FIGDIR = with_trailing_slash(expanduser(get(ENV, "FIGDIR", "/home/tsohail/uom/ocean-ensembles/figures/")))
const RESOLUTION = "sxtdeg"
const SECONDS_PER_YEAR = 365 * 24 * 60 * 60
const VIDEO_FRAMERATE = 12 # 12 frames per second for all videos
const TARGET_DEPTH_LEVELS = [75, 57, 37, 27, 17] # surface -> deeper
const PROGRESS_UPDATES = 20

function copy_files_to_tempdir(files::Vector{String}; prefix::String)
    copy_dir = mktempdir(; prefix)
    copied = String[]

    try
        for file in files
            dest = joinpath(copy_dir, basename(file))
            cp(file, dest; force = true)
            push!(copied, dest)
        end
    catch
        rm(copy_dir; recursive = true, force = true)
        rethrow()
    end

    @info "Copied analysis inputs." source_files = length(files) copy_dir
    return with_trailing_slash(copy_dir), copied
end

function cleanup_copied_outputs!(copy_dir::Union{Nothing, String})
    if copy_dir !== nothing && isdir(copy_dir)
        rm(copy_dir; recursive = true, force = true)
        @info "Deleted copied analysis inputs." copy_dir
    end
    return nothing
end

const VAR_TITLES = Dict(
    "T" => "Temperature (degC)",
    "S" => "Salinity (g/kg)",
    "u" => "Zonal Velocity (m/s)",
    "v" => "Meridional Velocity (m/s)",
    "speed" => "Horizontal Speed (m/s)",
    "ice_thickness" => "Ice Thickness (m)",
    "ice_concentration" => "Ice Concentration (%)",
    "u_ice" => "Zonal Velocity (m/s)",
    "v_ice" => "Meridional Velocity (m/s)",
)

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

function depth_slice_files(path::AbstractString)
    files = glob("global_*_fields_$(RESOLUTION)_RYF_run*.jld2", path)
    files = filter(files) do f
        !occursin("_rank", f) && occursin(r"global_\d+_fields_", f) && run_id(f) >= 0
    end
    sort!(files; by = run_id)
    return files
end

function sea_ice_surface_files(path::AbstractString)
    files = glob("global_sea_ice_surface_$(RESOLUTION)_RYF_run*.jld2", path)
    files = filter(files) do f
        !occursin("_rank", f) && run_id(f) >= 0
    end
    sort!(files; by = run_id)
    return files
end

function unique_depth_levels(files::Vector{String})
    levels = Int[]
    for f in files
        m = match(r"global_(\d+)_fields_", basename(f))
        m === nothing && continue
        push!(levels, parse(Int, m.captures[1]))
    end
    sort!(unique(levels))
    return levels
end

function selected_depth_levels(files::Vector{String})
    available = Set(unique_depth_levels(files))
    selected = [d for d in TARGET_DEPTH_LEVELS if d in available]
    isempty(selected) && error("None of TARGET_DEPTH_LEVELS=$(TARGET_DEPTH_LEVELS) found in depth files.")
    return selected
end

function unique_iterations(files::Vector{String})
    runs = Int[]
    for f in files
        r = run_id(f)
        r >= 0 && push!(runs, r)
    end
    sort!(unique(runs))
    return runs
end

function load_grid_from_output_file(filepath::AbstractString)
    grid = jldopen(filepath, "r") do f
        haskey(f, "serialized/grid") ? f["serialized/grid"] : nothing
    end

    if grid === nothing
        prefix = replace(filepath, r"\.jld2$" => "")
        @info "Falling back to create_grid for legacy layout." filepath
        return create_grid(prefix; gridtype = "TripolarGrid")
    end

    return grid
end

function load_depth_variable_timeseries(var::String, depths::Vector{Int}, iterations::Vector{Int}; path::AbstractString = OUTPUT_PATH)
    all_depth_times = Vector{Vector{Float64}}()
    all_depth_data = Vector{Vector{Matrix{Float32}}}()
    is_speed = var == "speed"
    @info "Loading depth timeseries..." variable = var depth_count = length(depths) iteration_count = length(iterations)
    depth_progress_step = max(1, cld(length(depths), PROGRESS_UPDATES))

    for (depth_index, depth) in enumerate(depths)
        records_by_time = Dict{Float64, NamedTuple{(:run, :data), Tuple{Int, Matrix{Float32}}}}()
        replaced_duplicates = 0
        iter_progress_step = max(1, cld(length(iterations), PROGRESS_UPDATES))

        for (iter_index, iteration) in enumerate(iterations)
            run = lpad(string(iteration), 4, '0')
            filepath = joinpath(path, "global_$(depth)_fields_$(RESOLUTION)_RYF_run$(run).jld2")
            isfile(filepath) || continue

            jldopen(filepath, "r") do f
                haskey(f, "timeseries/t") || return
                if is_speed
                    (haskey(f, "timeseries/u") && haskey(f, "timeseries/v")) || return
                else
                    haskey(f, "timeseries/$var") || return
                end

                ts_keys = sort(parse.(Int, collect(keys(f["timeseries/t"]))))
                for key in ts_keys
                    tval = Float64(f["timeseries/t/$key"])

                    A = if is_speed
                        u2 = extract_2d_f32(f["timeseries/u/$key"])
                        v2 = extract_2d_f32(f["timeseries/v/$key"])
                        if u2 === nothing || v2 === nothing
                            nothing
                        else
                            S = similar(u2, Float32)
                            @inbounds for i in eachindex(S, u2, v2)
                                S[i] = sqrt(u2[i]^2 + v2[i]^2)
                            end
                            S
                        end
                    else
                        extract_2d_f32(f["timeseries/$var/$key"])
                    end

                    A === nothing && continue
                    existing = get(records_by_time, tval, nothing)
                    if isnothing(existing) || iteration >= existing.run
                        replaced_duplicates += !isnothing(existing) && iteration > existing.run ? 1 : 0
                        records_by_time[tval] = (run = iteration, data = A)
                    end
                end
            end

            if iter_index == 1 || iter_index == length(iterations) || iter_index % iter_progress_step == 0
                log_record_progress("load_$(var)_depth$(depth)", iter_index, length(iterations))
            end
        end

        sorted_times = sort(collect(keys(records_by_time)))
        sorted_data = [records_by_time[t].data for t in sorted_times]
        push!(all_depth_times, sorted_times)
        push!(all_depth_data, sorted_data)
        @info "Loaded depth level." variable = var depth depth_index frames = length(sorted_data) replaced_duplicates
        if depth_index == 1 || depth_index == length(depths) || depth_index % depth_progress_step == 0
            log_record_progress("depth_levels_$(var)", depth_index, length(depths))
        end
    end

    @info "Completed depth timeseries load." variable = var total_depths = length(all_depth_data)
    return all_depth_times, all_depth_data
end

function load_surface_timeseries(files::Vector{String}, vars::Vector{String})
    records_by_time = Dict{Float64, NamedTuple{(:run, :fields), Tuple{Int, Vector{Matrix{Float32}}}}}()
    replaced_duplicates = 0
    @info "Loading surface timeseries..." file_count = length(files) variables = vars

    for (file_index, file) in enumerate(files)
        run = run_id(file)
        jldopen(file, "r") do f
            has_t = haskey(f, "timeseries/t")
            missing = [v for v in vars if !haskey(f, "timeseries/$v")]
            if !has_t || !isempty(missing)
                @warn "Skipping sea-ice file: required fields missing." file missing
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
                    replaced_duplicates += !isnothing(existing) && run > existing.run ? 1 : 0
                    records_by_time[tval] = (run = run, fields = timestep_fields)
                end
            end
        end

        if file_index == 1 || file_index == length(files) || file_index % max(1, cld(length(files), PROGRESS_UPDATES)) == 0
            log_record_progress("surface_files", file_index, length(files))
        end
    end

    all_time = sort(collect(keys(records_by_time)))
    all_data = Dict{String, Vector{Matrix{Float32}}}(v => Matrix{Float32}[] for v in vars)
    for (var_idx, var) in enumerate(vars)
        all_data[var] = [records_by_time[t].fields[var_idx] for t in all_time]
    end
    @info "Completed surface timeseries load." frames = length(all_time) variables = vars replaced_duplicates

    return all_time, all_data
end

function nearest_time_indices(target_times::Vector{Float64}, source_times::Vector{Float64})
    isempty(source_times) && error("Cannot align times with an empty source_times vector.")
    idx = Vector{Int}(undef, length(target_times))

    for (n, t) in enumerate(target_times)
        j = searchsortedfirst(source_times, t)
        if j <= 1
            idx[n] = 1
        elseif j > length(source_times)
            idx[n] = length(source_times)
        else
            left = j - 1
            right = j
            idx[n] = abs(source_times[left] - t) <= abs(source_times[right] - t) ? left : right
        end
    end

    return idx
end

function log_record_progress(tag::String, frame::Int, nframes::Int)
    width = 24
    fraction = frame / nframes
    filled = clamp(floor(Int, width * fraction), 0, width)
    bar = "[" * repeat("=", filled) * repeat(".", width - filled) * "]"
    percent = round(100 * fraction; digits = 1)
    @info "Recording progress." variable = tag frame nframes percent bar
end

function depth_color_settings(var::String, all_depth_data::Vector{Vector{Matrix{Float32}}})
    if var == "S"
        return :viridis, (34.8f0, 35.7f0)
    elseif var in ("u", "v")
        return :balance, (-0.5f0, 0.5f0)
    elseif var == "speed"
        return :speed, (0f0, 0.7f0)
    else
        A0 = all_depth_data[1][end]
        return :viridis, (minimum(A0), maximum(A0))
    end
end

function sea_ice_color_settings(var::String, A::Matrix{Float32})
    if var == "u_ice" || var == "v_ice"
        return :balance, (-0.5f0, 0.5f0)
    elseif var == "ice_concentration"
        vmax = maximum(A)
        return :ice, vmax <= 1.2f0 ? (0f0, 1f0) : (0f0, 100f0)
    elseif var == "ice_thickness"
        return :ice, (0f0, max(1f0, maximum(A)))
    else
        return :viridis, (minimum(A), maximum(A))
    end
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

function make_depth_variable_video(var::String,
                                   depths::Vector{Int},
                                   depths_actual::Vector{Float64},
                                   iterations::Vector{Int};
                                   output_path::AbstractString = OUTPUT_PATH,
                                   outname::Union{Nothing, String} = nothing,
                                   sea_ice_files::Vector{String} = String[],
                                   overlay_surface_ice::Bool = false,
                                   framerate::Int = VIDEO_FRAMERATE)
    all_depth_times, all_depth_data = load_depth_variable_timeseries(var, depths, iterations; path = output_path)
    any(isempty, all_depth_data) && error("At least one depth has zero frames for $var.")

    nframes_by_depth = length.(all_depth_data)
    nframes = minimum(nframes_by_depth)
    nframes > 0 || error("No frames available for $var.")
    if any(n != nframes for n in nframes_by_depth)
        @warn "Depth frame counts differ; truncating to shortest series." variable = var nframes_by_depth nframes
    end

    nd = length(depths)
    ncols = min(3, nd)
    nrows = cld(nd, ncols)
    times = all_depth_times[1][1:nframes]
    years = times ./ SECONDS_PER_YEAR

    cmap, clim = depth_color_settings(var, all_depth_data)
    @info "Depth color settings selected." variable = var colormap = cmap colorrange = clim
    Z = [Observable(all_depth_data[d][1]) for d in 1:nd]
    overlay_ice = nothing
    empty_ice_overlay = fill(NaN32, size(all_depth_data[1][1]))

    if overlay_surface_ice
        if isempty(sea_ice_files)
            @warn "Sea-ice files were not provided; skipping ice overlay on $(var) depth animation."
        else
            ice_times, ice_data = load_surface_timeseries(sea_ice_files, ["ice_thickness"])
            ice_frames = length(ice_data["ice_thickness"])
            if ice_frames == 0
                @warn "No ice thickness frames found; skipping ice overlay on $(var) depth animation."
            else
                idx = nearest_time_indices(times, ice_times)
                overlay_ice = Vector{Union{Nothing, Matrix{Float32}}}(undef, nframes)

                positive_dts = filter(>(0), diff(ice_times))
                max_mismatch = isempty(positive_dts) ? 0.0 : 0.51 * minimum(positive_dts)
                has_ice = false

                for frame in 1:nframes
                    ii = idx[frame]
                    t = times[frame]
                    δt = abs(ice_times[ii] - t)
                    in_time_range = first(ice_times) <= t <= last(ice_times)
                    close_enough = max_mismatch == 0.0 ? (δt == 0.0) : (δt <= max_mismatch)

                    if in_time_range && close_enough
                        overlay_ice[frame] = ice_data["ice_thickness"][ii]
                        has_ice = true
                    else
                        overlay_ice[frame] = nothing
                    end
                end

                if !has_ice
                    @warn "No matching ice timesteps found for depth frames; skipping ice overlay on $(var) depth animation."
                    overlay_ice = nothing
                else
                    first_ice_frame = findfirst(!isnothing, overlay_ice)
                    ice_first = overlay_ice[first_ice_frame]
                    if size(ice_first) != size(all_depth_data[1][1])
                        @warn "Ice thickness shape does not match depth field shape; skipping ice overlay." depth_size = size(all_depth_data[1][1]) ice_size = size(ice_first)
                        overlay_ice = nothing
                    end
                end
            end
        end
    end

    fig = Figure(size = (550 * ncols, 320 * nrows + 120))
    hms = Heatmap[]
    Zice = nothing
    hm_ice = nothing

    for k in 1:nd
        i = cld(k, ncols)
        j = (k - 1) % ncols + 1
        ax = Axis(fig[i, j], title = "Depth $(round(depths_actual[k], digits=1)) m")
        hm = heatmap!(ax, Z[k], colormap = cmap, colorrange = clim)
        if k == 1 && overlay_ice !== nothing
            available_ice = filter(!isnothing, overlay_ice)
            initial_ice = isnothing(overlay_ice[1]) ? empty_ice_overlay : overlay_ice[1]
            Zice = Observable(initial_ice)
            clim_ice = (0f0, max(1f0, Float32(maximum(maximum, available_ice))))
            hm_ice = heatmap!(ax, Zice, colormap = :ice, colorrange = clim_ice, alpha = 0.45)
        end
        push!(hms, hm)
    end

    fig_title = Label(fig[0, :], "Loading...", tellwidth = false)
    Colorbar(fig[nrows + 1, :], hms[1], label = get(VAR_TITLES, var, var), vertical = false)
    if hm_ice !== nothing
        Colorbar(fig[1, ncols + 1], hm_ice, label = get(VAR_TITLES, "ice_thickness", "ice_thickness"))
    end
    resize_to_layout!(fig)

    isnothing(outname) && (outname = FIGDIR * "$(var)_$(RESOLUTION)_all_depths.mp4")

    @info "Recording depth animation..." variable = var outname nframes framerate
    progress_step = max(1, cld(nframes, 20))
    record(fig, outname, 1:nframes; framerate = framerate) do frame
        fig_title.text = "$(get(VAR_TITLES, var, var)) | Year = $(round(years[frame], digits = 2))"
        for d in 1:nd
            Z[d][] = all_depth_data[d][frame]
        end
        if Zice !== nothing
            Zice[] = isnothing(overlay_ice[frame]) ? empty_ice_overlay : overlay_ice[frame]
        end
        if frame == 1 || frame == nframes || frame % progress_step == 0
            log_record_progress(var, frame, nframes)
        end
    end

    @info "Saved depth animation." variable = var outname
    return outname
end

function make_sea_ice_surface_polar_animation(files::Vector{String},
                                              grid;
                                              hemisphere::Symbol = :north,
                                              latitude_cutoff::Real = 50,
                                              vars::Vector{String} = ["ice_thickness", "ice_concentration", "u_ice", "v_ice"],
                                              outname::Union{Nothing, String} = nothing,
                                              framerate::Int = VIDEO_FRAMERATE)
    times, all_data = load_surface_timeseries(files, vars)
    nframes = minimum(length.(values(all_data)))
    nframes > 0 || error("No sea-ice surface frames found for requested variables.")
    if any(length(all_data[v]) != nframes for v in vars)
        @warn "Sea-ice variables have inconsistent frame counts; truncating to shortest." nframes
    end

    lon, lat = center_lon_lat(grid)
    cutoff = Float32(latitude_cutoff)
    mask = hemisphere === :north ? lat .>= cutoff : lat .<= -cutoff
    any(mask) || error("No points found for hemisphere=$(hemisphere) with cutoff=$(latitude_cutoff).")

    source_proj = "+proj=longlat +datum=WGS84"
    dest_proj = stereographic_projection(hemisphere)

    observables = Dict{String, Observable{Matrix{Float32}}}()
    for var in vars
        size(all_data[var][1]) == size(lon) ||
            error("Sea-ice field size mismatch for $var: got $(size(all_data[var][1])) expected $(size(lon)).")
        observables[var] = Observable(ifelse.(mask, all_data[var][1], NaN32))
    end

    hemi_title = hemisphere === :north ? "Arctic" : "Southern Ocean"
    fig = Figure(size = (1400, 900))
    fig_title = Label(fig[0, :], "$(hemi_title) loading...", tellwidth = false)

    axs = GeoAxis[]
    for (panel_idx, var) in enumerate(vars)
        row = cld(panel_idx, 2)
        col = (panel_idx - 1) % 2 + 1
        ax = GeoAxis(fig[row, col];
                     source = source_proj,
                     dest = dest_proj,
                     title = "$(get(VAR_TITLES, var, var)) ($(hemi_title))")
        push!(axs, ax)
        hidedecorations!(ax)
        hidespines!(ax)
        cmap, clim = sea_ice_color_settings(var, all_data[var][1])
        hm = surface!(ax,
                      lon,
                      lat,
                      zeros(Float32, size(lon));
                      color = observables[var],
                      shading = NoShading,
                      colormap = cmap,
                      colorrange = clim)
        Colorbar(fig[row + 2, col], hm, vertical = false)
    end

    for ax in axs
        xlims!(ax, -180, 180)
        if hemisphere === :north
            ylims!(ax, latitude_cutoff, 90)
        else
            ylims!(ax, -90, -latitude_cutoff)
        end
    end

    resize_to_layout!(fig)
    if isnothing(outname)
        hemi_suffix = hemisphere === :north ? "arctic" : "southern_ocean"
        outname = FIGDIR * "sea_ice_surface_$(RESOLUTION)_$(hemi_suffix)_all_runs.mp4"
    end

    years = times[1:nframes] ./ SECONDS_PER_YEAR
    @info "Recording sea-ice animation..." hemisphere outname nframes framerate
    progress_step = max(1, cld(nframes, 20))
    record(fig, outname, 1:nframes; framerate = framerate) do frame
        fig_title.text = "$(hemi_title) sea-ice surface fields | Year = $(round(years[frame], digits = 2))"
        for var in vars
            observables[var][] = ifelse.(mask, all_data[var][frame], NaN32)
        end
        if frame == 1 || frame == nframes || frame % progress_step == 0
            log_record_progress("sea_ice_$(Symbol(hemisphere))", frame, nframes)
        end
    end

    @info "Saved sea-ice polar animation." hemisphere outname
    return outname
end

function run_all_animations()
    @info "Starting horizontal analysis animations." output_path = OUTPUT_PATH resolution = RESOLUTION

    copy_output_path = nothing

    try
        live_depth_files = depth_slice_files(OUTPUT_PATH)
        isempty(live_depth_files) && error("No depth-slice files found in $(OUTPUT_PATH).")

        live_sea_ice_files = sea_ice_surface_files(OUTPUT_PATH)
        copy_output_path, _ = copy_files_to_tempdir(unique(vcat(live_depth_files, live_sea_ice_files));
                                                    prefix = "analysis2d_horizontal_")

        depth_files = depth_slice_files(copy_output_path)
        sea_ice_files = sea_ice_surface_files(copy_output_path)

        grid_file = first(depth_files)
        grid = load_grid_from_output_file(grid_file)
        @info "Loaded grid for animations." grid_file

        depths = selected_depth_levels(depth_files)
        runs = unique_iterations(depth_files)
        depths_actual = abs.(grid.z.cᵃᵃᶠ[depths])

        @info "Prepared animation inputs." depth_files = length(depth_files) runs = length(runs) depths sea_ice_files = length(sea_ice_files)

        for var in ("T", "S", "u", "v", "speed")
            @info "Processing depth variable..." variable = var
            make_depth_variable_video(var, depths, depths_actual, runs;
                                      output_path = copy_output_path,
                                      sea_ice_files = sea_ice_files,
                                      overlay_surface_ice = (var == "T" || var == "S" || var == "speed"),
                                      outname = FIGDIR * "$(var)_$(RESOLUTION)_all_depths.mp4")
        end

        if isempty(sea_ice_files)
            @warn "No sea-ice surface files found. Skipping sea-ice animation."
        else
            make_sea_ice_surface_polar_animation(sea_ice_files, grid;
                                                 hemisphere = :north,
                                                 latitude_cutoff = 50,
                                                 outname = FIGDIR * "sea_ice_surface_$(RESOLUTION)_arctic_all_runs.mp4")

            make_sea_ice_surface_polar_animation(sea_ice_files, grid;
                                                 hemisphere = :south,
                                                 latitude_cutoff = 50,
                                                 outname = FIGDIR * "sea_ice_surface_$(RESOLUTION)_southern_ocean_all_runs.mp4")
        end
    finally
        cleanup_copied_outputs!(copy_output_path)
    end

    @info "Completed all horizontal animations."
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_animations()
end
