using CairoMakie
using JLD2
using Glob

const OUTPUT_PATH = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
const FIGDIR = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")
const RESOLUTION = "sxtdeg"
const SECONDS_PER_YEAR = 365 * 24 * 60 * 60
const COLOR_SIGMA_MULTIPLE = 3.0
const MAX_COLOR_SAMPLES = 1_000_000
const MAX_COLOR_FRAMES = 240
const PROGRESS_UPDATES = 20
const forcing_timesteps_days = Float64[]
const SWAPPABLE_PAIRS = Dict(
    :surface_tracers => ["T_surf", "S_surf"],
    :fluxes => ["fw_flux", "heat_flux"])
# Current default: surface height + T_surf + S_surf.
# To switch later, set e.g. ACTIVE_PAIR = :fluxes
const ACTIVE_PAIR = :fluxes
const VAR_TITLES = Dict(
    "surface_height" => "Surface Height (m)",
    "heat_flux" => "Total Heat Flux (W m⁻²)",
    "fw_flux" => "Total Mass Flux (kg m⁻² s⁻¹)",
    "ocean_heat_flux" => "Ocean Heat Flux (W m⁻²)",
    "ocean_freshwater_flux" => "Ocean Mass Flux (kg m⁻² s⁻¹)", 
    "sea_ice_heat_flux" => "Sea Ice Heat Flux (W m⁻²)", 
    "sea_ice_freshwater_flux" => "Sea Ice Mass Flux (kg m⁻² s⁻¹)"
)

@inline function report_progress_step(i::Int, total::Int; label::AbstractString)
    stride = max(1, cld(total, PROGRESS_UPDATES))
    if i == 1 || i == total || i % stride == 0
        pct = round(100 * i / total; digits = 1)
        @info label progress = "$(i)/$(total)" percent = pct
    end
    return nothing
end

function run_id(path::AbstractString)
    m = match(r"run(\d+)", basename(path))
    return m === nothing ? -1 : parse(Int, m.captures[1])
end

function forcing_files(path::AbstractString)
    files = glob("global_surface_fluxes_$(RESOLUTION)*_RYF_run*.jld2", path)
    files = filter(files) do f
        !occursin("_rank", f) && occursin("surface_fluxes", f) && run_id(f) >= 0
    end
    sort!(files; by = run_id)
    return files
end

struct FrameRef
    file::String
    key::Int
    time::Float64
    run::Int
end

mutable struct RunningStats
    n::Int
    mean::Float64
    m2::Float64
end

RunningStats() = RunningStats(0, 0.0, 0.0)

@inline function extract_2d(raw)
    if ndims(raw) == 2
        return raw
    elseif ndims(raw) == 3
        return view(raw, :, :, 1)
    end
    return nothing
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

function collect_frame_refs(files::Vector{String}, vars::Vector{String})
    isempty(files) && error("No forcing files found in $OUTPUT_PATH.")
    frame_by_time = Dict{Float64, FrameRef}()
    used_files = 0
    replaced_duplicates = 0

    @info "Collecting frame references" file_count = length(files)

    for (file_index, file) in enumerate(files)
        run = run_id(file)
        @info "Scanning file" file_index total_files = length(files) file
        jldopen(file, "r") do f
            haskey(f, "timeseries/t") || return
            ts_available = filter(k -> k != "t", collect(keys(f["timeseries"])))
            missing = setdiff(vars, ts_available)
            if !isempty(missing)
                @warn "Skipping file because required variables are missing." file missing
                return
            end

            ts_keys = sort(parse.(Int, collect(keys(f["timeseries/t"]))) )
            for key in ts_keys
                tval = Float64(f["timeseries/t/$key"])
                candidate = FrameRef(file, key, tval, run)
                existing = get(frame_by_time, tval, nothing)
                if isnothing(existing) || run > existing.run || (run == existing.run && key >= existing.key)
                    replaced_duplicates += !isnothing(existing) && run > existing.run ? 1 : 0
                    frame_by_time[tval] = candidate
                end
            end
            used_files += 1
        end

        report_progress_step(file_index, length(files); label = "File scan")
    end

    used_files == 0 && error("No valid forcing files contained all required variables: $(join(vars, ", ")).")
    isempty(frame_by_time) && error("No timesteps found in forcing files.")
    frames = collect(values(frame_by_time))
    sort!(frames; by = frame -> frame.time)
    @info "Finished collecting frame references" valid_files = used_files frames = length(frames) replaced_duplicates
    return frames
end

function allocate_frame_buffers(vars::Vector{String}, first_frame::FrameRef)
    buffers = Dict{String, Matrix{Float32}}()
    jldopen(first_frame.file, "r") do f
        for var in vars
            raw = f["timeseries/$var/$(first_frame.key)"]
            src = extract_2d(raw)
            src === nothing && error("Unsupported array rank for variable=$var in $(first_frame.file), key=$(first_frame.key).")
            A = Matrix{Float32}(undef, size(src)...)
            copy_2d_to!(A, raw) || error("Failed to initialize frame buffer for variable=$var.")
            buffers[var] = A
        end
    end
    @info "Allocated frame buffers" vars dimensions = Dict(var => size(buffers[var]) for var in keys(buffers))
    return buffers
end

@inline function update_stats!(stats::RunningStats, A::Matrix{Float32}, stride::Int)
    @inbounds for i in 1:stride:length(A)
        x = Float64(A[i])
        isfinite(x) || continue
        stats.n += 1
        δ = x - stats.mean
        stats.mean += δ / stats.n
        stats.m2 += δ * (x - stats.mean)
    end
end

function sampled_colormap_limits(frames::Vector{FrameRef}, vars::Vector{String}, buffers::Dict{String, Matrix{Float32}})
    frame_step = max(1, cld(length(frames), MAX_COLOR_FRAMES))
    sampled_count = cld(length(frames), frame_step)

    @info "Sampling colormap limits" sampled_frames = sampled_count total_frames = length(frames) frame_step

    stats = Dict{String, RunningStats}(var => RunningStats() for var in vars)
    strides = Dict{String, Int}()
    for var in vars
        total_values = sampled_count * length(buffers[var])
        strides[var] = max(1, cld(total_values, MAX_COLOR_SAMPLES))
    end

    current_file = ""
    handle = nothing
    sample_indices = 1:frame_step:length(frames)

    try
        for (sample_index, i) in enumerate(sample_indices)
            frame = frames[i]
            if frame.file != current_file
                handle !== nothing && close(handle)
                handle = jldopen(frame.file, "r")
                current_file = frame.file
            end

            for var in vars
                ok = copy_2d_to!(buffers[var], handle["timeseries/$var/$(frame.key)"])
                ok || error("Inconsistent array shape for variable=$var in $(frame.file), key=$(frame.key).")
                update_stats!(stats[var], buffers[var], strides[var])
            end

            report_progress_step(sample_index, sampled_count; label = "Colormap sampling")
        end
    finally
        handle !== nothing && close(handle)
    end

    limits = Dict{String, Tuple{Symbol, Tuple{Float32, Float32}}}()
    for var in vars
        s = stats[var]
        if s.n == 0
            @warn "All sampled values were non-finite for variable; using fallback colorrange." variable = var
            limits[var] = (:balance, (-1f0, 1f0))
            continue
        end

        σ = s.n > 1 ? sqrt(s.m2 / (s.n - 1)) : 0.0
        σ = max(σ, eps(Float64))
        kσ = max(Float32(COLOR_SIGMA_MULTIPLE * σ), 1f-6)
        limits[var] = (:balance, (-kσ, kσ))
    end

    @info "Finished colormap limit sampling"
    return limits
end

function load_frame!(buffers::Dict{String, Matrix{Float32}}, file, vars::Vector{String}, key::Int)
    for var in vars
        ok = copy_2d_to!(buffers[var], file["timeseries/$var/$key"])
        ok || error("Inconsistent array shape for variable=$var at key=$key.")
    end
end

function make_forcing_animation(; outname = FIGDIR * "forcing_fields_$(RESOLUTION)_all_runs.mp4", framerate = 6)
    @info "Starting forcing animation build" output = outname framerate

    files = forcing_files(OUTPUT_PATH)
    @info "Using forcing files" count = length(files)

    haskey(SWAPPABLE_PAIRS, ACTIVE_PAIR) || error("ACTIVE_PAIR=$(ACTIVE_PAIR) not found. Valid options: $(join(string.(collect(keys(SWAPPABLE_PAIRS))), ", "))")
    selected_vars = vcat(SWAPPABLE_PAIRS[ACTIVE_PAIR])
    @info "Selected variables" selected_vars

    frames = collect_frame_refs(files, selected_vars)
    buffers = allocate_frame_buffers(selected_vars, frames[1])
    colormap_limits = sampled_colormap_limits(frames, selected_vars, buffers)

    nframes = length(frames)
    @info "Preparing figure and render loop" nframes

    fig = Figure(size = (1800, 700))
    title = Label(fig[0, :], "Loading...", tellwidth = false)

    observables = Dict{String, Observable{Matrix{Float32}}}()
    for (i, var) in enumerate(selected_vars)
        ax = Axis(fig[1, i], title = get(VAR_TITLES, var, var))
        observables[var] = Observable(copy(buffers[var]))
        cmap, clim = colormap_limits[var]
        hm = heatmap!(ax, observables[var], colormap = cmap, colorrange = clim)
        Colorbar(fig[2, i], hm, vertical = false)
    end
    resize_to_layout!(fig)

    timesteps_days = [frame.time for frame in frames] ./ (24 * 3600)
    empty!(forcing_timesteps_days)
    append!(forcing_timesteps_days, timesteps_days)
    years = timesteps_days ./ 365

    current_file = Ref("")
    handle = Ref{Any}(nothing)
    @info "Starting MP4 render" outname

    try
        record(fig, outname, 1:nframes; framerate) do frame_index
            frame = frames[frame_index]
            if frame.file != current_file[]
                handle[] !== nothing && close(handle[])
                handle[] = jldopen(frame.file, "r")
                current_file[] = frame.file
            end

            load_frame!(buffers, handle[], selected_vars, frame.key)
            title.text = "Global surface forcing fields (divergent scale, ±1σ) | Run $(frame.run) | Year = $(round(years[frame_index], digits=2))"
            for var in selected_vars
                copyto!(observables[var][], buffers[var])
                notify(observables[var])
            end

            report_progress_step(frame_index, nframes; label = "Frame render")
        end
    finally
        handle[] !== nothing && close(handle[])
    end

    @info "Saved animation" outname nframes
    @info "Forcing timesteps (days)" forcing_timesteps_days
    return outname
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_forcing_animation()
end
