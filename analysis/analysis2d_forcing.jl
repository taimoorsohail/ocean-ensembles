using CairoMakie
using JLD2
using Logging
using Glob
using Oceananigans
using NumericalEarth

const OUTPUT_PATH = expanduser("/home/tsohail/uom/ocean-ensembles/outputs/saved/")
const FIGDIR = expanduser("/home/tsohail/uom/ocean-ensembles/figures/")
const RESOLUTION = "sxtdeg"
const SECONDS_PER_YEAR = 365 * 24 * 60 * 60
const COLOR_SIGMA_MULTIPLE = 3.0
const MAX_COLOR_SAMPLES = 1_000_000
const MAX_COLOR_FRAMES = 240
const PROGRESS_UPDATES = 20
const GC_INTERVAL = 12
const DEFAULT_COLORRANGE = (0f0, 1f0)
const NAN_PLOT_COLOR = :lightgray
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
    files = glob("combined_global_surface_fluxes_$(RESOLUTION)*_RYF_run*.jld2", path)
    files = filter(files) do f
        !occursin("_rank", f) && occursin("surface_fluxes", f) && run_id(f) >= 0
    end
    sort!(files; by = run_id)
    return files
end

function ssh_files(path::AbstractString)
    files = glob("combined_global_surface_fluxes_$(RESOLUTION)_RYF_run*.jld2", path)
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

underlying_grid(grid) = hasproperty(grid, :underlying_grid) ? getproperty(grid, :underlying_grid) : grid

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

            if sample_index % GC_INTERVAL == 0
                GC.gc(false)
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
            @warn "No finite values found for colormap limits; using default range." variable = var default = DEFAULT_COLORRANGE
            limits[var] = (:balance, DEFAULT_COLORRANGE)
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

function bottom_height_matrix(filepath::AbstractString)
    return with_logger(NullLogger()) do
        jldopen(filepath, "r") do f
            haskey(f, "serialized/grid") || return nothing
            grid = f["serialized/grid"]
            hasproperty(grid, :immersed_boundary) || return nothing

            source_grid = underlying_grid(grid)
            immersed_boundary = getproperty(grid, :immersed_boundary)
            hasproperty(immersed_boundary, :bottom_height) || return nothing

            bottom_height_field = getproperty(immersed_boundary, :bottom_height)
            hasproperty(bottom_height_field, :data) || return nothing
            physical_matrix(getproperty(bottom_height_field, :data), source_grid; T = Float32)
        end
    end
end

surface_ocean_mask(bottom_height::Union{Nothing, AbstractMatrix}) = isnothing(bottom_height) ? nothing : bottom_height .< 0f0

function mask_field_with_plot_mask!(dest::Matrix{Float32}, src::Matrix{Float32}, plot_mask::Union{Nothing, AbstractMatrix{Bool}})
    size(dest) == size(src) || error("Destination/src size mismatch: $(size(dest)) vs $(size(src)).")
    isnothing(plot_mask) && return copyto!(dest, src)
    size(dest) == size(plot_mask) || error("Plot mask shape mismatch: got $(size(plot_mask)) expected $(size(dest)).")
    @inbounds for i in eachindex(dest, src, plot_mask)
        dest[i] = plot_mask[i] ? src[i] : NaN32
    end
    return dest
end

function make_forcing_animation(; outname = FIGDIR * "forcing_fields_$(RESOLUTION)_all_runs.mp4", framerate = 6)
    @info "Starting forcing animation build" output = outname framerate

    files = forcing_files(OUTPUT_PATH)
    @info "Using forcing files in place" count = length(files)

        haskey(SWAPPABLE_PAIRS, ACTIVE_PAIR) || error("ACTIVE_PAIR=$(ACTIVE_PAIR) not found. Valid options: $(join(string.(collect(keys(SWAPPABLE_PAIRS))), ", "))")
        selected_vars = vcat(SWAPPABLE_PAIRS[ACTIVE_PAIR])
        @info "Selected variables" selected_vars

        frames = collect_frame_refs(files, selected_vars)
        buffers = allocate_frame_buffers(selected_vars, frames[1])
        bottom_height = bottom_height_matrix(frames[1].file)
        surface_mask = surface_ocean_mask(bottom_height)
        colormap_limits = sampled_colormap_limits(frames, selected_vars, buffers)

        nframes = length(frames)
        @info "Preparing figure and render loop" nframes

        fig = Figure(size = (1800, 700))
        title = Label(fig[0, :], "Loading...", tellwidth = false)

        observables = Dict{String, Observable{Matrix{Float32}}}()
        for (i, var) in enumerate(selected_vars)
            ax = Axis(fig[1, i], title = get(VAR_TITLES, var, var))
            initial = copy(buffers[var])
            mask_field_with_plot_mask!(initial, buffers[var], surface_mask)
            observables[var] = Observable(initial)
            cmap, clim = colormap_limits[var]
            hm = heatmap!(ax, observables[var], colormap = cmap, colorrange = clim, nan_color = NAN_PLOT_COLOR)
            Colorbar(fig[2, i], hm, vertical = false)
        end
        resize_to_layout!(fig)

        empty!(forcing_timesteps_days)
        sizehint!(forcing_timesteps_days, nframes)
        for frame in frames
            push!(forcing_timesteps_days, frame.time / (24 * 3600))
        end

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
                year = frame.time / SECONDS_PER_YEAR
                title.text = "Global surface forcing fields (divergent scale, ±1σ) | Run $(frame.run) | Year = $(round(year, digits=2))"
                for var in selected_vars
                    mask_field_with_plot_mask!(observables[var][], buffers[var], surface_mask)
                    notify(observables[var])
                end

                if frame_index % GC_INTERVAL == 0
                    GC.gc(false)
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

function make_ssh_animation(; outname = FIGDIR * "ssh_fields_$(RESOLUTION)_all_runs.mp4", framerate = 6)
    @info "Starting SSH animation build" output = outname framerate

    files = ssh_files(OUTPUT_PATH)
    @info "Using SSH files in place" count = length(files)

        selected_vars = ["surface_height"]
        frames = collect_frame_refs(files, selected_vars)
        buffers = allocate_frame_buffers(selected_vars, frames[1])
        bottom_height = bottom_height_matrix(frames[1].file)
        surface_mask = surface_ocean_mask(bottom_height)
        colormap_limits = sampled_colormap_limits(frames, selected_vars, buffers)

        nframes = length(frames)
        @info "Preparing SSH figure and render loop" nframes

        fig = Figure(size = (900, 700))
        title = Label(fig[0, :], "Loading...", tellwidth = false)

        ssh = "surface_height"
        ax = Axis(fig[1, 1], title = get(VAR_TITLES, ssh, ssh))
        initial = copy(buffers[ssh])
        mask_field_with_plot_mask!(initial, buffers[ssh], surface_mask)
        observable = Observable(initial)
        cmap, clim = colormap_limits[ssh]
        hm = heatmap!(ax, observable, colormap = cmap, colorrange = clim, nan_color = NAN_PLOT_COLOR)
        Colorbar(fig[2, 1], hm, vertical = false)
        resize_to_layout!(fig)


        current_file = Ref("")
        handle = Ref{Any}(nothing)
        @info "Starting SSH MP4 render" outname

        try
            record(fig, outname, 1:nframes; framerate) do frame_index
                frame = frames[frame_index]
                if frame.file != current_file[]
                    handle[] !== nothing && close(handle[])
                    handle[] = jldopen(frame.file, "r")
                    current_file[] = frame.file
                end

                load_frame!(buffers, handle[], selected_vars, frame.key)
                year = frame.time / SECONDS_PER_YEAR
                title.text = "Global SSH (divergent scale, ±1σ) | Run $(frame.run) | Year = $(round(year, digits=2))"
                mask_field_with_plot_mask!(observable[], buffers[ssh], surface_mask)
                notify(observable)

                if frame_index % GC_INTERVAL == 0
                    GC.gc(false)
                end
                report_progress_step(frame_index, nframes; label = "SSH frame render")
            end
        finally
            handle[] !== nothing && close(handle[])
        end

    @info "Saved SSH animation" outname nframes
    return outname
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_forcing_animation()
    make_ssh_animation()
end
