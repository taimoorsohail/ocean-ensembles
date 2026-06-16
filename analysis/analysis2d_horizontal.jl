using CairoMakie
using JLD2
using Glob
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
const DEPTH_FILE_RUN_PREFIX = "_fields_$(RESOLUTION)_RYF_run"
const DEFAULT_COLORRANGE = (0f0, 1f0)
const NAN_PLOT_COLOR = :lightgray

const VAR_TITLES = Dict(
    "T" => "Temperature (degC)",
    "S" => "Salinity (g/kg)",
    "u" => "Zonal Velocity (m/s)",
    "v" => "Meridional Velocity (m/s)",
    "w" => "Vertical Velocity (m/s)",
    "du_dt" => "du/dt (m/s^2)",
    "dv_dt" => "dv/dt (m/s^2)",
    "dw_dt" => "dw/dt (m/s^2)",
    "speed" => "Horizontal Speed (m/s)",
)

pretty_var_name(var::String) = get(VAR_TITLES, var, replace(var, "_" => " "))

@inline function run_id(path::AbstractString)
    m = match(r"run(\d+)", basename(path))
    return m === nothing ? -1 : parse(Int, m.captures[1])
end

function validate_selected_run(selected_run::Union{Nothing, Int})
    isnothing(selected_run) && return nothing
    selected_run < 0 && throw(ArgumentError("Run number must be non-negative, got $(selected_run)."))
    return selected_run
end

function parse_selected_run()
    isempty(ARGS) && return nothing
    any(arg -> arg in ("-h", "--help"), ARGS) && return :help
    length(ARGS) == 1 || throw(ArgumentError("Expected at most one positional argument: run number."))
    selected_run = tryparse(Int, ARGS[1])
    selected_run === nothing && throw(ArgumentError("Could not parse run number from \"$(ARGS[1])\"."))
    return validate_selected_run(selected_run)
end

function print_usage()
    println("Usage: julia --project=ocean-ensembles/analysis ocean-ensembles/analysis/analysis2d_horizontal.jl [run_id]")
    println("If run_id is provided, only files from that run are used when building animations.")
    return nothing
end

run_suffix(run::Int) = "run" * lpad(string(run), 4, '0')

function filter_files_by_run(files::Vector{String}, selected_run::Union{Nothing, Int})
    isnothing(selected_run) && return files
    return filter(f -> run_id(f) == selected_run, files)
end

@inline function extract_2d_f32(raw)
    if ndims(raw) == 2
        return Float32.(raw)
    elseif ndims(raw) == 3
        return Float32.(raw[:, :, 1])
    end
    return nothing
end

function speed_matrix_or_nothing(u::AbstractMatrix, v::AbstractMatrix; context::AbstractString)
    if size(u) != size(v)
        @warn "Skipping speed frame because u and v shapes differ." context u_size = size(u) v_size = size(v)
        return nothing
    end

    S = similar(u, Float32)
    @inbounds for i in eachindex(S, u, v)
        S[i] = sqrt(u[i]^2 + v[i]^2)
    end
    return S
end

function depth_slice_files(path::AbstractString)
    files = glob("combined_*$(DEPTH_FILE_RUN_PREFIX)*.jld2", path)
    files = filter(files) do f
        !occursin("_rank", f) && occursin(r"global_\d+_fields_", basename(f)) && run_id(f) >= 0
    end
    sort!(files; by = run_id)
    return files
end


function top_surface_files(path::AbstractString)
    files = glob("combined_*surface*.jld2", path)
    files = filter(files) do f
        !occursin("_rank", f) &&
        !occursin("surface_fluxes", basename(f))
    end

    if isempty(files)
        files = glob("combined_*$(DEPTH_FILE_RUN_PREFIX)*.jld2", path)
        files = filter(files) do f
            !occursin("_rank", f)
        end
    end

    sort!(files)
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

function file_has_timeseries_layout(filepath::AbstractString)
    return jldopen(filepath, "r") do f
        haskey(f, "timeseries/t")
    end
end

function discover_timeseries_variables(filepath::AbstractString)
    return jldopen(filepath, "r") do f
        haskey(f, "timeseries") || return String[]
        vars = String[]
        for key in keys(f["timeseries"])
            key == "t" && continue
            push!(vars, String(key))
        end
        sort!(vars)
        if "u" in vars && "v" in vars && !("speed" in vars)
            push!(vars, "speed")
        end
        return vars
    end
end

function discover_top_level_matrix_variables(filepath::AbstractString)
    return jldopen(filepath, "r") do f
        vars = String[]
        for key in keys(f)
            name = String(key)
            name in ("time", "Δt", "surface_level") && continue
            raw = f[name]
            if raw isa AbstractArray && ndims(raw) in (2, 3)
                push!(vars, name)
            end
        end
        sort!(vars)
        if "u" in vars && "v" in vars && !("speed" in vars)
            push!(vars, "speed")
        end
        return vars
    end
end

function discover_animation_variables(files::Vector{String})
    isempty(files) && return String[]
    first_file = first(files)
    return file_has_timeseries_layout(first_file) ? discover_timeseries_variables(first_file) : discover_top_level_matrix_variables(first_file)
end


function load_top_level_timeseries(files::Vector{String}, vars::Vector{String})
    records_by_time = Dict{Float64, NamedTuple{(:file_index, :fields), Tuple{Int, Dict{String, Matrix{Float32}}}}}()
    @info "Loading top-level timeseries..." file_count = length(files) variables = vars

    for (file_index, file) in enumerate(files)
        jldopen(file, "r") do f
            haskey(f, "time") || return
            tval = Float64(f["time"])
            timestep_fields = Dict{String, Matrix{Float32}}()

            for var in vars
                if var == "speed"
                    (haskey(f, "u") && haskey(f, "v")) || return
                    u = extract_2d_f32(f["u"])
                    v = extract_2d_f32(f["v"])
                    (u === nothing || v === nothing) && return
                    S = speed_matrix_or_nothing(u, v; context = "top-level file $(basename(file)) at time $(tval)")
                    S === nothing && return
                    timestep_fields[var] = S
                else
                    haskey(f, var) || return
                    A = extract_2d_f32(f[var])
                    A === nothing && return
                    timestep_fields[var] = A
                end
            end

            existing = get(records_by_time, tval, nothing)
            if isnothing(existing) || file_index >= existing.file_index
                records_by_time[tval] = (file_index = file_index, fields = timestep_fields)
            end
        end

        if file_index == 1 || file_index == length(files) || file_index % max(1, cld(length(files), PROGRESS_UPDATES)) == 0
            log_record_progress("top_level_files", file_index, length(files))
        end
    end

    all_time = sort(collect(keys(records_by_time)))
    all_data = Dict{String, Vector{Matrix{Float32}}}(v => Matrix{Float32}[] for v in vars)
    for var in vars
        all_data[var] = [records_by_time[t].fields[var] for t in all_time]
    end

    @info "Completed top-level timeseries load." frames = length(all_time) variables = vars
    return all_time, all_data
end

function load_general_surface_timeseries(files::Vector{String}, vars::Vector{String})
    isempty(files) && error("No files were provided.")
    return file_has_timeseries_layout(first(files)) ? load_surface_timeseries(files, vars) : load_top_level_timeseries(files, vars)
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
            @info "No serialized grid found; skipping grid load for top-level file layout." filepath
            return nothing
        end
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
            filepath = joinpath(path, "combined_global_$(depth)$(DEPTH_FILE_RUN_PREFIX)$(run).jld2")
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
                            speed_matrix_or_nothing(u2, v2; context = "depth $(depth) run $(run) key $(key)")
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
                @warn "Skipping surface-timeseries file: required fields missing." file missing
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

function finite_colorrange_from_values(values; default = DEFAULT_COLORRANGE, context::AbstractString = "field")
    lo = Inf32
    hi = -Inf32
    found_finite = false

    for value in values
        if isfinite(value)
            value32 = Float32(value)
            lo = min(lo, value32)
            hi = max(hi, value32)
            found_finite = true
        end
    end

    if !found_finite
        @warn "No finite values found for colorrange; using default range." context default
        return default
    elseif !(hi > lo)
        pad = max(1f-6, 0.05f0 * max(abs(lo), 1f0))
        return (lo - pad, hi + pad)
    end

    return (lo, hi)
end

function finite_maximum_from_values(values; default = 0f0, context::AbstractString = "field")
    maxval = -Inf32
    found_finite = false

    for value in values
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

function depth_color_settings(var::String, all_depth_data::Vector{Vector{Matrix{Float32}}})
    if var == "S"
        return :blues, (34.8f0, 37f0)
    elseif var in ("u", "v")
        return :balance, (-0.5f0, 0.5f0)
    elseif var == "w"
        return :balance, (-2e-5, 2e-5)
    elseif var == "e"
        return :viridis, (0f0, 0.0015f0)
    elseif var == "speed"
        return :speed, (0f0, 0.7f0)
    else
        return :viridis, finite_colorrange_from_values((x for depth_data in all_depth_data for A in depth_data for x in A);
                                                        context = "depth variable $(var)")
    end
end


function make_depth_variable_video(var::String,
                                   depths::Vector{Int},
                                   depths_actual::Vector,
                                   iterations::Vector{Int};
                                   output_path::AbstractString = OUTPUT_PATH,
                                   bottom_height = nothing,
                                   outname::Union{Nothing, String} = nothing,
                                   framerate::Int = VIDEO_FRAMERATE)
    all_depth_times, all_depth_data = load_depth_variable_timeseries(var, depths, iterations; path = output_path)
    any(!isempty, all_depth_times) || error("No depth-timeseries frames found for variable=$(var).")

    reference_index = findfirst(!isempty, all_depth_times)
    reference_times = all_depth_times[reference_index]
    nframes = length(reference_times)
    alignment = [isempty(times) ? Int[] : nearest_time_indices(reference_times, times) for times in all_depth_times]
    depth_masks = [depth_ocean_mask(depth_value, bottom_height) for depth_value in depths_actual]
    cmap, clim = depth_color_settings(var, all_depth_data)

    initial_fields = Matrix{Float32}[]
    for depth_index in eachindex(depths)
        data = all_depth_data[depth_index]
        isempty(data) && error("No frames found for variable=$(var) at depth index $(depths[depth_index]).")
        push!(initial_fields, mask_field_with_plot_mask(data[alignment[depth_index][1]], depth_masks[depth_index]))
    end

    npanels = length(depths)
    ncols = min(3, npanels)
    nrows = cld(npanels, ncols)
    fig = Figure(size = (500 * ncols, 380 * nrows))
    title = Label(fig[0, :], "Loading...", tellwidth = false)
    observables = Observable.(initial_fields)

    for (panel_index, depth_index) in enumerate(eachindex(depths))
        row = cld(panel_index, ncols)
        col = mod1(panel_index, ncols)
        depth_label = round(abs(Float64(depths_actual[depth_index])); digits = 1)
        ax = Axis(fig[row, col], title = "Depth $(depth_label) m")
        hm = heatmap!(ax, observables[depth_index], colormap = cmap, colorrange = clim, nan_color = NAN_PLOT_COLOR)
        Colorbar(fig[row + nrows, col], hm, vertical = false)
    end
    resize_to_layout!(fig)

    outname = isnothing(outname) ? FIGDIR * "$(var)_$(RESOLUTION)_all_depths.mp4" : outname
    years = reference_times ./ SECONDS_PER_YEAR
    progress_step = max(1, cld(nframes, PROGRESS_UPDATES))

    record(fig, outname, 1:nframes; framerate = framerate) do frame
        title.text = "$(pretty_var_name(var)) | Year = $(round(years[frame], digits = 2))"
        for depth_index in eachindex(depths)
            source_index = alignment[depth_index][frame]
            observables[depth_index][] = mask_field_with_plot_mask(all_depth_data[depth_index][source_index], depth_masks[depth_index])
        end
        if frame == 1 || frame == nframes || frame % progress_step == 0
            log_record_progress("depth_$(var)", frame, nframes)
        end
    end

    @info "Saved depth animation." variable = var outname nframes
    return outname
end

function make_top_surface_variable_video(var::String,
                                         files::Vector{String};
                                         bottom_height = nothing,
                                         outname::Union{Nothing, String} = nothing,
                                         framerate::Int = VIDEO_FRAMERATE)
    times, all_data = load_general_surface_timeseries(files, [var])
    nframes = length(times)
    nframes > 0 || error("No top-surface frames found for variable=$(var).")

    fields = all_data[var]
    surface_mask = surface_ocean_mask(bottom_height)
    cmap, clim = var == "speed" ?
                 (:speed, (0f0, max(0.7f0, finite_maximum_from_values((x for A in fields for x in A); default = 0.7f0, context = "surface speed")))) :
                 (:viridis, finite_colorrange_from_values((x for A in fields for x in A); context = "surface variable $(var)"))

    fig = Figure(size = (1000, 760))
    title = Label(fig[0, :], "Loading...", tellwidth = false)
    ax = Axis(fig[1, 1], title = pretty_var_name(var))
    observable = Observable(mask_field_with_plot_mask(fields[1], surface_mask))
    hm = heatmap!(ax, observable, colormap = cmap, colorrange = clim, nan_color = NAN_PLOT_COLOR)
    Colorbar(fig[2, 1], hm, vertical = false)
    resize_to_layout!(fig)

    outname = isnothing(outname) ? FIGDIR * "$(var)_$(RESOLUTION)_top_surface.mp4" : outname
    years = times ./ SECONDS_PER_YEAR
    progress_step = max(1, cld(nframes, PROGRESS_UPDATES))

    record(fig, outname, 1:nframes; framerate = framerate) do frame
        title.text = "$(pretty_var_name(var)) | Year = $(round(years[frame], digits = 2))"
        observable[] = mask_field_with_plot_mask(fields[frame], surface_mask)
        if frame == 1 || frame == nframes || frame % progress_step == 0
            log_record_progress("surface_$(var)", frame, nframes)
        end
    end

    @info "Saved top-surface animation." variable = var outname nframes
    return outname
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

function bottom_height_matrix(grid)
    isnothing(grid) && return nothing
    hasproperty(grid, :immersed_boundary) || return nothing

    immersed_boundary = getproperty(grid, :immersed_boundary)
    hasproperty(immersed_boundary, :bottom_height) || return nothing

    bottom_height_field = getproperty(immersed_boundary, :bottom_height)
    bottom_height = Array(interior(bottom_height_field, :, :, 1))
    return Float32.(bottom_height)
end

function mask_field_with_plot_mask(A::AbstractMatrix, plot_mask::Union{Nothing, AbstractMatrix{Bool}})
    masked = Float32.(A)
    isnothing(plot_mask) && return masked
    size(masked) == size(plot_mask) || error("Plot mask shape mismatch: got $(size(plot_mask)) expected $(size(masked)).")
    @inbounds for i in eachindex(masked, plot_mask)
        plot_mask[i] || (masked[i] = NaN32)
    end
    return masked
end

surface_ocean_mask(bottom_height::Union{Nothing, AbstractMatrix}) = isnothing(bottom_height) ? nothing : bottom_height .< 0f0

depth_ocean_mask(depth::Real, bottom_height::Union{Nothing, AbstractMatrix}) =
    isnothing(bottom_height) ? nothing : bottom_height .< -Float32(depth)

function run_all_animations(selected_run::Union{Nothing, Int} = parse_selected_run())
    selected_run === :help && return print_usage()
    selected_run = validate_selected_run(selected_run)

    @info "Starting horizontal analysis animations." output_path = OUTPUT_PATH resolution = RESOLUTION

    depth_files = filter_files_by_run(depth_slice_files(OUTPUT_PATH), selected_run)
    top_surface_only_files = filter_files_by_run(top_surface_files(OUTPUT_PATH), selected_run)

    if !isnothing(selected_run) &&
       isempty(depth_files) &&
       isempty(top_surface_only_files)
        error("No animation inputs found for requested run $(run_suffix(selected_run)) in $(OUTPUT_PATH).")
    end

    if isempty(depth_files) && isempty(top_surface_only_files)
        error("No depth-slice or top-surface files found in $(OUTPUT_PATH).")
    end

    grid_file = !isempty(depth_files) ? first(depth_files) : first(top_surface_only_files)
    grid = load_grid_from_output_file(grid_file)
    isnothing(grid) || @info "Loaded grid for animations." grid_file
    bottom_height = isnothing(grid) ? nothing : bottom_height_matrix(grid)

    output_suffix = isnothing(selected_run) ? "all_runs" : run_suffix(selected_run)

    if !isempty(depth_files)
        depths = selected_depth_levels(depth_files)
        runs = isnothing(selected_run) ? unique_iterations(depth_files) : [selected_run]
        depths_actual = abs.(grid.z.cᵃᵃᶠ[depths])

        @info "Prepared depth animation inputs." depth_files = length(depth_files) runs = length(runs) depths selected_run

        depth_vars = discover_animation_variables(depth_files)
        @info "Discovered depth animation variables." variables = depth_vars
        for var in depth_vars
            (var == "speed" || !occursin("_dt", var)) || continue
            var in ("u", "v") && continue
            @info "Processing depth variable..." variable = var
            make_depth_variable_video(var, depths, depths_actual, runs;
                                      output_path = OUTPUT_PATH,
                                      bottom_height = bottom_height,
                                      outname = FIGDIR * "$(var)_$(RESOLUTION)_all_depths_$(output_suffix).mp4")
        end
    else
        @info "No depth-coded files found; falling back to top-surface plotting." top_surface_files = length(top_surface_only_files)
        surface_vars = discover_animation_variables(top_surface_only_files)
        @info "Discovered top-surface animation variables." variables = surface_vars
        for var in surface_vars
            var in ("u", "v") && continue
            @info "Processing top-surface variable..." variable = var
            make_top_surface_variable_video(var, top_surface_only_files;
                                            bottom_height = bottom_height,
                                            outname = FIGDIR * "$(var)_$(RESOLUTION)_top_surface_$(output_suffix).mp4")
        end
    end

    @info "Completed all horizontal animations."
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_animations()
end

