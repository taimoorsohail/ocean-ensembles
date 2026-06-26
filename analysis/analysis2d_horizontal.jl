using CairoMakie
using JLD2
using Glob
using Statistics: median
using Oceananigans
using Oceananigans.Fields: location
using Oceananigans.BoundaryConditions: fill_halo_regions!

with_trailing_slash(path) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator

const OUTPUT_PATH = with_trailing_slash(expanduser(get(ENV, "OUTPUT_PATH", "/home/tsohail/uom/ocean-ensembles/outputs/saved/")))
const FIGDIR = with_trailing_slash(expanduser(get(ENV, "FIGDIR", "/home/tsohail/uom/ocean-ensembles/figures/")))
const RESOLUTION = "sxtdeg"
const SECONDS_PER_YEAR = 365 * 24 * 60 * 60
const VIDEO_FRAMERATE = 12
const VIDEO_SIM_YEARS_PER_SECOND = 0.2
const MIN_VIDEO_FRAMERATE = 1.0
const MAX_VIDEO_FRAMERATE = 60.0
const TARGET_DEPTH_LEVELS = [75, 57, 37, 27, 17] # surface -> deeper
const PROGRESS_UPDATES = 20
const GC_INTERVAL = 12
const VIDEO_CHUNK_SIZE = 120
const DEPTH_FILE_RUN_PREFIX = "_fields_$(RESOLUTION)_RYF_run"
const DEFAULT_COLORRANGE = (0f0, 1f0)
const NAN_PLOT_COLOR = :lightgray
const DEFAULT_ANIMATION_VARS = ["T", "S", "e", "speed", "w"]
const DEPTH_COLORRANGE_STD_MULTIPLIER = parse(Float32, get(ENV, "HORIZONTAL_DEPTH_COLORRANGE_STD_MULTIPLIER", "1.0"))

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

@inline function extract_2d(raw)
    if ndims(raw) == 2
        return raw
    elseif ndims(raw) == 3
        return view(raw, :, :, 1)
    end
    return nothing
end

@inline function extract_2d_f32(raw)
    src = extract_2d(raw)
    src === nothing && return nothing
    return Float32.(src)
end

function copy_2d_to!(dest::Matrix{Float32}, raw)
    src = extract_2d(raw)
    src === nothing && return false
    size(dest) == size(src) || return false

    @inbounds for i in eachindex(dest, src)
        dest[i] = Float32(src[i])
    end
    return true
end

function speed_matrix_or_nothing(u::AbstractMatrix, v::AbstractMatrix; context::AbstractString)
    if size(u) != size(v)
        @warn "Skipping speed frame because u and v shapes differ and no grid metadata was available." context u_size = size(u) v_size = size(v)
        return nothing
    end

    S = similar(u, Float32)
    @inbounds for i in eachindex(S, u, v)
        S[i] = sqrt(u[i]^2 + v[i]^2)
    end
    return S
end

@inline surface_matrix_3d(A::AbstractMatrix) = reshape(A, size(A, 1), size(A, 2), 1)

function speed_workspace(grid)
    ufield = XFaceField(grid)
    vfield = YFaceField(grid)
    speed_field = @at (Center, Center, Nothing) sqrt(ufield^2 + vfield^2) |> Field
    return (; ufield, vfield, speed_field)
end

function centered_speed_matrix_or_nothing(raw_u, raw_v, grid; context::AbstractString, workspace = nothing)
    u2 = extract_2d_f32(raw_u)
    v2 = extract_2d_f32(raw_v)
    (u2 === nothing || v2 === nothing) && return nothing

    if isnothing(grid)
        return speed_matrix_or_nothing(u2, v2; context)
    end

    workspace = isnothing(workspace) ? speed_workspace(grid) : workspace
    set!(workspace.ufield, surface_matrix_3d(u2))
    set!(workspace.vfield, surface_matrix_3d(v2))
    fill_halo_regions!(workspace.ufield)
    fill_halo_regions!(workspace.vfield)
    compute!(workspace.speed_field)
    return Float32.(Array(interior(workspace.speed_field)[:, :, 1]))
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
        grid = load_grid_from_output_file(file)
        speed_cache = "speed" in vars && !isnothing(grid) ? speed_workspace(grid) : nothing

        jldopen(file, "r") do f
            haskey(f, "time") || return
            tval = Float64(f["time"])
            timestep_fields = Dict{String, Matrix{Float32}}()

            for var in vars
                if var == "speed"
                    (haskey(f, "u") && haskey(f, "v")) || return
                    S = centered_speed_matrix_or_nothing(f["u"], f["v"], grid; context = "top-level file $(basename(file)) at time $(tval)", workspace = speed_cache)
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

function index_depth_variable_timeseries(var::String, depths::Vector{Int}, iterations::Vector{Int}; path::AbstractString = OUTPUT_PATH)
    all_depth_times = Vector{Vector{Float64}}()
    all_depth_refs = Vector{Vector{NamedTuple{(:time, :run, :filepath, :key), Tuple{Float64, Int, String, Int}}}}()
    is_speed = var == "speed"
    @info "Indexing depth timeseries..." variable = var depth_count = length(depths) iteration_count = length(iterations)
    depth_progress_step = max(1, cld(length(depths), PROGRESS_UPDATES))

    for (depth_index, depth) in enumerate(depths)
        records_by_time = Dict{Float64, NamedTuple{(:run, :filepath, :key), Tuple{Int, String, Int}}}()
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
                    existing = get(records_by_time, tval, nothing)
                    if isnothing(existing) || iteration >= existing.run
                        replaced_duplicates += !isnothing(existing) && iteration > existing.run ? 1 : 0
                        records_by_time[tval] = (run = iteration, filepath = filepath, key = key)
                    end
                end
            end

            if iter_index == 1 || iter_index == length(iterations) || iter_index % iter_progress_step == 0
                log_record_progress("load_$(var)_depth$(depth)", iter_index, length(iterations))
            end
        end

        sorted_times = sort(collect(keys(records_by_time)))
        push!(all_depth_times, sorted_times)
        push!(all_depth_refs, [(time = t, run = records_by_time[t].run, filepath = records_by_time[t].filepath, key = records_by_time[t].key) for t in sorted_times])
        @info "Indexed depth level." variable = var depth depth_index frames = length(sorted_times) replaced_duplicates
        if depth_index == 1 || depth_index == length(depths) || depth_index % depth_progress_step == 0
            log_record_progress("depth_levels_$(var)", depth_index, length(depths))
        end
    end

    @info "Completed depth timeseries indexing." variable = var total_depths = length(all_depth_refs)
    return all_depth_times, all_depth_refs
end

function depth_frame_loader(var::String)
    is_speed = var == "speed"
    grid_cache = Dict{String, Any}()
    speed_cache = Dict{String, Any}()

    function load_frame(ref; context::AbstractString)
        filepath = ref.filepath

        grid = get!(grid_cache, filepath) do
            load_grid_from_output_file(filepath)
        end

        workspace = if is_speed && !isnothing(grid)
            get!(speed_cache, filepath) do
                speed_workspace(grid)
            end
        else
            nothing
        end

        return jldopen(filepath, "r") do f
            if is_speed
                (haskey(f, "timeseries/u") && haskey(f, "timeseries/v")) || return nothing
                centered_speed_matrix_or_nothing(f["timeseries/u/$(ref.key)"], f["timeseries/v/$(ref.key)"], grid; context, workspace = workspace)
            else
                haskey(f, "timeseries/$var") || return nothing
                extract_2d_f32(f["timeseries/$var/$(ref.key)"])
            end
        end
    end

    return load_frame
end

function depth_frame_reader(var::String)
    is_speed = var == "speed"
    current_file = Ref("")
    handle = Ref{Any}(nothing)
    grid_cache = Dict{String, Any}()
    speed_cache = Dict{String, Any}()

    function ensure_handle(filepath::String)
        if filepath != current_file[]
            handle[] !== nothing && close(handle[])
            handle[] = jldopen(filepath, "r")
            current_file[] = filepath
        end
        return handle[]
    end

    function load_frame!(dest::Matrix{Float32}, ref; context::AbstractString)
        filepath = ref.filepath
        file = ensure_handle(filepath)
        grid = get!(grid_cache, filepath) do
            load_grid_from_output_file(filepath)
        end

        if is_speed
            (haskey(file, "timeseries/u") && haskey(file, "timeseries/v")) || return false
            state = get!(speed_cache, filepath) do
                raw_u = file["timeseries/u/$(ref.key)"]
                raw_v = file["timeseries/v/$(ref.key)"]
                src_u = extract_2d(raw_u)
                src_v = extract_2d(raw_v)
                (src_u === nothing || src_v === nothing) && return nothing
                (; workspace = isnothing(grid) ? nothing : speed_workspace(grid),
                   u_buffer = Matrix{Float32}(undef, size(src_u)...),
                   v_buffer = Matrix{Float32}(undef, size(src_v)...))
            end
            isnothing(state) && return false
            copy_2d_to!(state.u_buffer, file["timeseries/u/$(ref.key)"]) || return false
            copy_2d_to!(state.v_buffer, file["timeseries/v/$(ref.key)"]) || return false

            if isnothing(grid)
                speed = speed_matrix_or_nothing(state.u_buffer, state.v_buffer; context)
                speed === nothing && return false
                size(dest) == size(speed) || return false
                copyto!(dest, speed)
                return true
            end

            set!(state.workspace.ufield, surface_matrix_3d(state.u_buffer))
            set!(state.workspace.vfield, surface_matrix_3d(state.v_buffer))
            fill_halo_regions!(state.workspace.ufield)
            fill_halo_regions!(state.workspace.vfield)
            compute!(state.workspace.speed_field)
            src = interior(state.workspace.speed_field)[:, :, 1]
            size(dest) == size(src) || return false
            @inbounds for i in eachindex(dest, src)
                dest[i] = Float32(src[i])
            end
            return true
        else
            haskey(file, "timeseries/$var") || return false
            return copy_2d_to!(dest, file["timeseries/$var/$(ref.key)"])
        end
    end

    function close_reader!()
        handle[] !== nothing && close(handle[])
        handle[] = nothing
        current_file[] = ""
        return nothing
    end

    return (; load_frame!, close_reader!)
end
function load_surface_timeseries(files::Vector{String}, vars::Vector{String})
    records_by_time = Dict{Float64, NamedTuple{(:run, :fields), Tuple{Int, Vector{Matrix{Float32}}}}}()
    replaced_duplicates = 0
    @info "Loading surface timeseries..." file_count = length(files) variables = vars

    for (file_index, file) in enumerate(files)
        run = run_id(file)
        grid = "speed" in vars ? load_grid_from_output_file(file) : nothing
        speed_cache = "speed" in vars && !isnothing(grid) ? speed_workspace(grid) : nothing

        jldopen(file, "r") do f
            has_t = haskey(f, "timeseries/t")
            missing = String[]
            for v in vars
                if v == "speed"
                    (haskey(f, "timeseries/u") && haskey(f, "timeseries/v")) || append!(missing, ["u", "v"])
                elseif !haskey(f, "timeseries/$v")
                    push!(missing, v)
                end
            end
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
                    A = if var == "speed"
                        centered_speed_matrix_or_nothing(f["timeseries/u/$key"], f["timeseries/v/$key"], grid; context = "surface file $(basename(file)) run $(run) key $(key)", workspace = speed_cache)
                    else
                        extract_2d_f32(f["timeseries/$var/$key"])
                    end
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

function constant_model_dt_framerate(times::Vector{Float64};
                                     fallback::Real = VIDEO_FRAMERATE,
                                     sim_years_per_second::Real = VIDEO_SIM_YEARS_PER_SECOND)
    length(times) > 1 || return Int(round(fallback))
    Δts = [times[i + 1] - times[i] for i in 1:length(times)-1 if times[i + 1] > times[i]]
    isempty(Δts) && return Int(round(fallback))

    fps = sim_years_per_second * SECONDS_PER_YEAR / median(Δts)
    return round(Int, clamp(fps, MIN_VIDEO_FRAMERATE, MAX_VIDEO_FRAMERATE))
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

function sampled_reference_indices(n::Int; max_samples::Int = 12)
    n <= 0 && return Int[]
    n <= max_samples && return collect(1:n)
    return unique(round.(Int, range(1, n; length = max_samples)))
end

function depth_color_settings_streaming(var::String,
                                        depth_refs::Vector{Vector{NamedTuple{(:time, :run, :filepath, :key), Tuple{Float64, Int, String, Int}}}},
                                        load_frame!;
                                        std_multiplier::Real = DEPTH_COLORRANGE_STD_MULTIPLIER)
    @info "Sampling depth frames to determine colorrange." variable = var depths = length(depth_refs) std_multiplier

    n = 0
    mean_value = 0.0
    m2 = 0.0
    found_finite = false
    for (depth_index, refs) in enumerate(depth_refs)
        sample_indices = sampled_reference_indices(length(refs))
        for ref_index in sample_indices
            ref = refs[ref_index]
            frame = load_frame!(ref; context = "colorrange $(var) $(basename(ref.filepath)) key $(ref.key)")
            isnothing(frame) && continue
            for value in frame
                if isfinite(value)
                    value64 = Float64(value)
                    n += 1
                    δ = value64 - mean_value
                    mean_value += δ / n
                    m2 += δ * (value64 - mean_value)
                    found_finite = true
                end
            end
        end
        if depth_index == 1 || depth_index == length(depth_refs) || depth_index % max(1, cld(length(depth_refs), PROGRESS_UPDATES)) == 0
            log_record_progress("colorrange_$(var)", depth_index, length(depth_refs))
        end
    end

    cmap = var == "T" ? :thermal :
           var == "S" ? :haline :
           var == "speed" ? :speed :
           var in ("u", "v", "w") ? :balance :
           :viridis

    if !found_finite
        default = var == "speed" ? (0f0, 0.7f0) : DEFAULT_COLORRANGE
        @warn "No finite sampled values found for colorrange; using default range." variable var default
        return cmap, default
    elseif n == 1
        center = Float32(mean_value)
        pad = 1f-6
        return cmap, var == "speed" ? (max(0f0, center - pad), center + pad) : (center - pad, center + pad)
    end

    std_value = sqrt(m2 / (n - 1))
    halfwidth = max(Float32(std_multiplier * std_value), 1f-6)
    center = Float32(mean_value)

    if var == "speed"
        return cmap, (max(0f0, center - halfwidth), max(center + halfwidth, 1f-6))
    end

    return cmap, (center - halfwidth, center + halfwidth)
end

function panel_layout(npanels::Int)
    npanels > 0 || error("At least one panel is required.")
    if npanels <= 3
        return 1, npanels
    elseif npanels == 4
        return 2, 2
    else
        return cld(npanels, 3), 3
    end
end

sanitize_var_token(var::String) = replace(var, r"[^A-Za-z0-9]+" => "-")
vars_slug(vars::Vector{String}) = join(sanitize_var_token.(vars), "_")

function resolve_ffmpeg_binary()
    ffmpeg = Sys.which("ffmpeg")
    !isnothing(ffmpeg) && return ffmpeg

    fallback = "/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/13.3.0/FFmpeg/7.0.2/bin/ffmpeg"
    return isfile(fallback) ? fallback : nothing
end

function record_video_in_chunks(render_frame!,
                                fig,
                                outname::AbstractString,
                                frames,
                                framerate::Int;
                                chunk_size::Int = VIDEO_CHUNK_SIZE,
                                tag::AbstractString = "video",
                                chunk_cleanup::Union{Nothing, Function} = nothing)
    isempty(frames) && error("No frames were provided for chunked recording.")
    mkpath(dirname(outname))

    ffmpeg = resolve_ffmpeg_binary()
    isnothing(ffmpeg) && error("Could not find ffmpeg for chunked video assembly.")

    chunk_dir = mktempdir(dirname(outname); prefix = basename(outname) * "_chunks_")
    chunk_paths = String[]

    try
        for (chunk_index, start_idx) in enumerate(1:chunk_size:length(frames))
            stop_idx = min(length(frames), start_idx + chunk_size - 1)
            chunk_frames = collect(frames[start_idx:stop_idx])
            chunk_path = joinpath(chunk_dir, "chunk_" * lpad(string(chunk_index), 4, '0') * ".mp4")
            push!(chunk_paths, chunk_path)
            @info "Recording video chunk." tag chunk_index start_idx stop_idx chunk_path
            record(fig, chunk_path, chunk_frames; framerate = framerate) do frame
                render_frame!(frame)
            end
            # Cleanup between chunks
            if !isnothing(chunk_cleanup)
                chunk_cleanup()
            end
            # Aggressively clear memory between chunks
            chunk_frames = nothing
            GC.gc(false)
            GC.gc()
            GC.gc()
        end

        concat_file = joinpath(chunk_dir, "concat.txt")
        open(concat_file, "w") do io
            for chunk_path in chunk_paths
                println(io, "file '" * replace(abspath(chunk_path), "'" => "'\''") * "'")
            end
        end

        cmd = `$ffmpeg -y -f concat -safe 0 -i $concat_file -c copy $outname`
        @info "Concatenating video chunks." tag outname chunk_count = length(chunk_paths) ffmpeg
        run(cmd)
    finally
        rm(chunk_dir; recursive = true, force = true)
    end

    return outname
end

function validate_requested_variables(requested_vars::Vector{String}, available_vars::Vector{String})
    isempty(requested_vars) && throw(ArgumentError("At least one variable must be requested."))
    missing = [var for var in requested_vars if !(var in available_vars)]
    isempty(missing) || error("Requested variables not available: $(missing). Available variables: $(available_vars)")
    return requested_vars
end

function single_depth_level(k::Union{Nothing, Int}, available_depths::Vector{Int})
    isempty(available_depths) && error("No depth levels are available.")
    isnothing(k) && return first(available_depths)
    k in available_depths || error("Requested depth level k=$(k) not available. Available levels: $(available_depths)")
    return k
end

function variable_color_settings(var::String, fields::Vector{Matrix{Float32}})
    if var == "T"
        return :thermal, (-2f0, 32f0)
    elseif var == "S"
        return :haline, (34.8f0, 37f0)
    elseif var in ("u", "v")
        return :balance, (-0.5f0, 0.5f0)
    elseif var == "w"
        return :balance, (-2e-5, 2e-5)
    elseif var == "e"
        return :viridis, (0f0, 0.0015f0)
    elseif var == "speed"
        return :speed, (0f0, max(0.7f0, finite_maximum_from_values((x for A in fields for x in A); default = 0.7f0, context = "variable $(var)")))
    else
        return :viridis, finite_colorrange_from_values((x for A in fields for x in A); context = "variable $(var)")
    end
end

function make_horizontal_slice_video(vars::Vector{String},
                                     depth::Int,
                                     depth_actual::Real,
                                     iterations::Vector{Int};
                                     output_path::AbstractString = OUTPUT_PATH,
                                     bottom_height = nothing,
                                     outname::Union{Nothing, String} = nothing,
                                     framerate::Union{Nothing, Real} = nothing)
    refs_by_var = Dict{String, Vector{NamedTuple{(:time, :run, :filepath, :key), Tuple{Float64, Int, String, Int}}}}()
    times_by_var = Dict{String, Vector{Float64}}()
    loaders = Dict{String, Function}()

    for var in vars
        depth_times, depth_refs = index_depth_variable_timeseries(var, [depth], iterations; path = output_path)
        isempty(depth_times) && error("No timeseries index returned for variable=$(var) at depth level k=$(depth).")
        isempty(depth_times[1]) && error("No depth-timeseries frames found for variable=$(var) at depth level k=$(depth).")
        times_by_var[var] = depth_times[1]
        refs_by_var[var] = depth_refs[1]
        loaders[var] = depth_frame_loader(var)
    end

    reference_var = first(vars)
    reference_times = times_by_var[reference_var]
    nframes = length(reference_times)
    alignments = Dict(var => nearest_time_indices(reference_times, times_by_var[var]) for var in vars)
    depth_mask = depth_ocean_mask(depth_actual, bottom_height)

    initial_fields = Dict{String, Matrix{Float32}}()
    color_settings = Dict{String, Tuple}()
    for var in vars
        load_frame! = loaders[var]
        initial_frame = load_frame!(refs_by_var[var][alignments[var][1]]; context = "initial $(var) depth $(depth)")
        initial_frame === nothing && error("Could not load initial frame for variable=$(var) at depth level k=$(depth).")
        initial_fields[var] = mask_field_with_plot_mask(initial_frame, depth_mask)
        color_settings[var] = depth_color_settings_streaming(var, [refs_by_var[var]], load_frame!)
    end
    frame_buffers = Dict(var => similar(initial_fields[var]) for var in vars)
    readers = Dict(var => depth_frame_reader(var) for var in vars)

    nrows, ncols = panel_layout(length(vars))
    fig = Figure(size = (480 * ncols, 360 * nrows + 80 * nrows))
    title = Label(fig[0, :], "Loading...", tellwidth = false)
    observables = Dict(var => Observable(initial_fields[var]) for var in vars)

    for (panel_index, var) in enumerate(vars)
        row = cld(panel_index, ncols)
        col = mod1(panel_index, ncols)
        layout_row = 2 * row - 1
        cmap, clim = color_settings[var]
        ax = Axis(fig[layout_row, col], title = pretty_var_name(var))
        hm = heatmap!(ax, observables[var], colormap = cmap, colorrange = clim, nan_color = NAN_PLOT_COLOR)
        Colorbar(fig[layout_row + 1, col], hm, vertical = false)
    end
    resize_to_layout!(fig)

    outname = isnothing(outname) ? FIGDIR * "horizontal_k$(depth)_$(vars_slug(vars))_$(RESOLUTION).mp4" : outname
    years = reference_times ./ SECONDS_PER_YEAR
    framerate = isnothing(framerate) ? constant_model_dt_framerate(reference_times) : framerate
    progress_step = max(1, cld(nframes, PROGRESS_UPDATES))

    @info "Recording horizontal slice animation." outname depth depth_actual variables = vars nframes framerate

    try
        record_video_in_chunks(fig, outname, 1:nframes, framerate; 
                              tag = "horizontal_k$(depth)",
                              chunk_cleanup = () -> begin
                                  # Close readers between chunks to release file handles
                                  for reader in values(readers)
                                      reader.close_reader!()
                                  end
                                  # Recreate readers for next chunk
                                  readers = Dict(var => depth_frame_reader(var) for var in vars)
                                  # Clear frame buffers
                                  for var in vars
                                      fill!(frame_buffers[var], NaN32)
                                  end
                              end) do frame
            title.text = "Depth k=$(depth) ($(round(abs(Float64(depth_actual)), digits = 1)) m) | Year = $(round(years[frame], digits = 2))"
            for var in vars
                ref = refs_by_var[var][alignments[var][frame]]
                ok = readers[var].load_frame!(frame_buffers[var], ref; context = "frame $(frame) $(var) depth $(depth)")
                ok || error("Could not load frame $(frame) for variable=$(var) at depth level k=$(depth).")
                mask_field_with_plot_mask!(observables[var][], frame_buffers[var], depth_mask)
                notify(observables[var])
            end
            if frame % GC_INTERVAL == 0
                GC.gc(false)
            end
            if frame % (5 * GC_INTERVAL) == 0
                GC.gc()
            end
            if frame == 1 || frame == nframes || frame % progress_step == 0
                log_record_progress("horizontal_k$(depth)", frame, nframes)
            end
        end
    finally
        for reader in values(readers)
            reader.close_reader!()
        end
    end

    @info "Saved horizontal slice animation." outname depth variables = vars nframes framerate
    return outname
end

function make_top_surface_multivariable_video(vars::Vector{String},
                                              files::Vector{String};
                                              bottom_height = nothing,
                                              outname::Union{Nothing, String} = nothing,
                                              framerate::Union{Nothing, Real} = nothing)
    times, all_data = load_general_surface_timeseries(files, vars)
    nframes = length(times)
    nframes > 0 || error("No top-surface frames found for variables=$(vars).")

    surface_mask = surface_ocean_mask(bottom_height)
    nrows, ncols = panel_layout(length(vars))
    fig = Figure(size = (480 * ncols, 360 * nrows + 80 * nrows))
    title = Label(fig[0, :], "Loading...", tellwidth = false)

    observables = Dict{String, Observable}()
    color_settings = Dict{String, Tuple}()
    for var in vars
        fields = all_data[var]
        observables[var] = Observable(mask_field_with_plot_mask(fields[1], surface_mask))
        color_settings[var] = variable_color_settings(var, fields)
    end

    for (panel_index, var) in enumerate(vars)
        row = cld(panel_index, ncols)
        col = mod1(panel_index, ncols)
        layout_row = 2 * row - 1
        cmap, clim = color_settings[var]
        ax = Axis(fig[layout_row, col], title = pretty_var_name(var))
        hm = heatmap!(ax, observables[var], colormap = cmap, colorrange = clim, nan_color = NAN_PLOT_COLOR)
        Colorbar(fig[layout_row + 1, col], hm, vertical = false)
    end
    resize_to_layout!(fig)

    outname = isnothing(outname) ? FIGDIR * "horizontal_surface_$(vars_slug(vars))_$(RESOLUTION).mp4" : outname
    years = times ./ SECONDS_PER_YEAR
    framerate = isnothing(framerate) ? constant_model_dt_framerate(times) : framerate
    progress_step = max(1, cld(nframes, PROGRESS_UPDATES))

    @info "Recording top-surface animation." outname variables = vars nframes framerate

    record(fig, outname, 1:nframes; framerate = framerate) do frame
        title.text = "Top surface | Year = $(round(years[frame], digits = 2))"
        for var in vars
            mask_field_with_plot_mask!(observables[var][], all_data[var][frame], surface_mask)
            notify(observables[var])
        end
        if frame % GC_INTERVAL == 0
            GC.gc(false)
        end
        if frame == 1 || frame == nframes || frame % progress_step == 0
            log_record_progress("surface_multi", frame, nframes)
        end
    end

    @info "Saved top-surface animation." outname variables = vars nframes framerate
    return outname
end


function make_depth_variable_video(var::String,
                                   depths::Vector{Int},
                                   depths_actual::Vector,
                                   iterations::Vector{Int};
                                   output_path::AbstractString = OUTPUT_PATH,
                                   bottom_height = nothing,
                                   outname::Union{Nothing, String} = nothing,
                                   framerate::Int = VIDEO_FRAMERATE)
    all_depth_times, all_depth_refs = index_depth_variable_timeseries(var, depths, iterations; path = output_path)
    any(!isempty, all_depth_times) || error("No depth-timeseries frames found for variable=$(var).")

    reference_index = findfirst(!isempty, all_depth_times)
    reference_times = all_depth_times[reference_index]
    nframes = length(reference_times)
    alignment = [isempty(times) ? Int[] : nearest_time_indices(reference_times, times) for times in all_depth_times]
    depth_masks = [depth_ocean_mask(depth_value, bottom_height) for depth_value in depths_actual]
    load_frame! = depth_frame_loader(var)
    color_settings = Vector{Tuple}(undef, length(depths))

    initial_fields = Matrix{Float32}[]
    for depth_index in eachindex(depths)
        refs = all_depth_refs[depth_index]
        isempty(refs) && error("No frames found for variable=$(var) at depth index $(depths[depth_index]).")
        frame_data = load_frame!(refs[alignment[depth_index][1]]; context = "initial $(var) depth $(depths[depth_index])")
        frame_data === nothing && error("Could not load initial frame for variable=$(var) at depth index $(depths[depth_index]).")
        push!(initial_fields, mask_field_with_plot_mask(frame_data, depth_masks[depth_index]))
        color_settings[depth_index] = depth_color_settings_streaming(var, [refs], load_frame!)
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
        cmap, clim = color_settings[depth_index]
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
            ref = all_depth_refs[depth_index][alignment[depth_index][frame]]
            frame_data = load_frame!(ref; context = "frame $(frame) $(var) depth $(depths[depth_index])")
            frame_data === nothing && error("Could not load frame $(frame) for variable=$(var) at depth index $(depths[depth_index]).")
            observables[depth_index][] = mask_field_with_plot_mask(frame_data, depth_masks[depth_index])
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

function mask_field_with_plot_mask!(dest::Matrix{Float32}, src::AbstractMatrix, plot_mask::Union{Nothing, AbstractMatrix{Bool}})
    size(dest) == size(src) || error("Destination/src size mismatch: $(size(dest)) vs $(size(src)).")
    if isnothing(plot_mask)
        copyto!(dest, src)
        return dest
    end

    size(dest) == size(plot_mask) || error("Plot mask shape mismatch: got $(size(plot_mask)) expected $(size(dest)).")
    @inbounds for i in eachindex(dest, src, plot_mask)
        dest[i] = plot_mask[i] ? Float32(src[i]) : NaN32
    end
    return dest
end

surface_ocean_mask(bottom_height::Union{Nothing, AbstractMatrix}) = isnothing(bottom_height) ? nothing : bottom_height .< 0f0

depth_ocean_mask(depth::Real, bottom_height::Union{Nothing, AbstractMatrix}) =
    isnothing(bottom_height) ? nothing : bottom_height .< -Float32(depth)

function run_all_animations(selected_run::Union{Nothing, Int} = parse_selected_run())
    return run_all_animations(DEFAULT_ANIMATION_VARS; selected_run = selected_run)
end

function run_all_animations(vars::Vector{String}; k::Union{Nothing, Int} = nothing, selected_run = parse_selected_run())
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
        available_depths = unique_depth_levels(depth_files)
        requested_depth = single_depth_level(k, available_depths)
        runs = isnothing(selected_run) ? unique_iterations(depth_files) : [selected_run]
        available_vars = filter(var -> var != "u" && var != "v" && (var == "speed" || !occursin("_dt", var)),
                                discover_animation_variables(depth_files))
        validate_requested_variables(vars, available_vars)
        isnothing(grid) && error("A grid is required to map depth level k=$(requested_depth) to physical depth.")
        depth_actual = abs(grid.z.cᵃᵃᶠ[requested_depth])

        @info "Prepared depth animation inputs." depth_files = length(depth_files) runs = length(runs) requested_depth depth_actual selected_run variables = vars
        make_horizontal_slice_video(vars, requested_depth, depth_actual, runs;
                                    output_path = OUTPUT_PATH,
                                    bottom_height = bottom_height,
                                    outname = FIGDIR * "horizontal_k$(requested_depth)_$(vars_slug(vars))_$(RESOLUTION)_$(output_suffix).mp4")
    else
        @info "No depth-coded files found; falling back to top-surface plotting." top_surface_files = length(top_surface_only_files)
        available_vars = filter(var -> var != "u" && var != "v", discover_animation_variables(top_surface_only_files))
        validate_requested_variables(vars, available_vars)
        make_top_surface_multivariable_video(vars, top_surface_only_files;
                                             bottom_height = bottom_height,
                                             outname = FIGDIR * "horizontal_surface_$(vars_slug(vars))_$(RESOLUTION)_$(output_suffix).mp4")
    end

    @info "Completed all horizontal animations."
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_animations()
end

