using CairoMakie
using JLD2
using Glob
using OceanEnsembles
using Oceananigans
using Oceananigans.Fields: location

const OUTPUT_PATH = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved_fields/onedeg/")
const FIGDIR = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")
const RESOLUTION = "onedeg"
const SECONDS_PER_YEAR = 365 * 24 * 60 * 60
const TARGET_DEPTH_LEVELS = [75, 57, 37, 27, 17] # surface -> deeper

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

function load_depth_variable_timeseries(var::String, depths::Vector{Int}, iterations::Vector{Int})
    all_depth_times = Vector{Vector{Float64}}()
    all_depth_data = Vector{Vector{Matrix{Float32}}}()
    is_speed = var == "speed"

    for depth in depths
        raw_times = Float64[]
        raw_data = Matrix{Float32}[]

        for iteration in iterations
            run = lpad(string(iteration), 4, '0')
            filepath = OUTPUT_PATH * "global_$(depth)_fields_$(RESOLUTION)_RYF_run$(run).jld2"
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
                    push!(raw_times, tval)
                    push!(raw_data, A)
                end
            end
        end

        order = sortperm(raw_times)
        push!(all_depth_times, raw_times[order])
        push!(all_depth_data, raw_data[order])
    end

    @info "Completed depth timeseries load." variable = var total_depths = length(all_depth_data)
    return all_depth_times, all_depth_data
end

function load_surface_timeseries(files::Vector{String}, vars::Vector{String})
    all_data = Dict{String, Vector{Matrix{Float32}}}(v => Matrix{Float32}[] for v in vars)
    all_time = Float64[]

    for file in files
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
                push!(all_time, tval)
                for (idx, var) in enumerate(vars)
                    push!(all_data[var], timestep_fields[idx])
                end
            end
        end
    end

    order = sortperm(all_time)
    all_time = all_time[order]
    for var in vars
        all_data[var] = all_data[var][order]
    end

    return all_time, all_data
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

function stereographic_xy(lon_deg, lat_deg; hemisphere::Symbol = :north, lon0_deg::Real = 0)
    λ = deg2rad.(lon_deg .- lon0_deg)
    φ = deg2rad.(lat_deg)

    sinφ = sin.(φ)
    cosφ = cos.(φ)
    ϵ = eps(Float32)

    if hemisphere === :north
        k = 2f0 ./ (1f0 .+ sinφ .+ ϵ)
        x = k .* cosφ .* sin.(λ)
        y = -k .* cosφ .* cos.(λ)
    elseif hemisphere === :south
        k = 2f0 ./ (1f0 .- sinφ .+ ϵ)
        x = k .* cosφ .* sin.(λ)
        y = k .* cosφ .* cos.(λ)
    else
        throw(ArgumentError("`hemisphere` must be :north or :south."))
    end

    return x, y
end

function polar_region_geometry(grid; hemisphere::Symbol = :north, latitude_cutoff::Real = 50)
    lon, lat = center_lon_lat(grid)
    cutoff = Float32(latitude_cutoff)

    mask = hemisphere === :north ? lat .>= cutoff : lat .<= -cutoff
    idx = findall(vec(mask))
    isempty(idx) && error("No points found for hemisphere=$(hemisphere) with cutoff=$(latitude_cutoff).")

    x2d, y2d = stereographic_xy(lon, lat; hemisphere)
    xv = vec(x2d)[idx]
    yv = vec(y2d)[idx]

    return idx, xv, yv
end

function make_depth_variable_video(var::String,
                                   depths::Vector{Int},
                                   depths_actual::Vector{Float64},
                                   iterations::Vector{Int};
                                   outname::Union{Nothing, String} = nothing,
                                   framerate::Int = 6)
    all_depth_times, all_depth_data = load_depth_variable_timeseries(var, depths, iterations)
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

    cmap, clim = depth_color_settings(var, all_depth_data)
    @info "Depth color settings selected." variable = var colormap = cmap colorrange = clim
    Z = [Observable(all_depth_data[d][1]) for d in 1:nd]

    fig = Figure(size = (550 * ncols, 320 * nrows + 120))
    hms = Heatmap[]

    for k in 1:nd
        i = cld(k, ncols)
        j = (k - 1) % ncols + 1
        ax = Axis(fig[i, j], title = "Depth $(round(depths_actual[k], digits=1)) m")
        hm = heatmap!(ax, Z[k], colormap = cmap, colorrange = clim)
        push!(hms, hm)
    end

    fig_title = Label(fig[0, :], "Loading...", tellwidth = false)
    Colorbar(fig[nrows + 1, :], hms[1], label = get(VAR_TITLES, var, var), vertical = false)
    resize_to_layout!(fig)

    isnothing(outname) && (outname = FIGDIR * "$(var)_$(RESOLUTION)_all_depths.mp4")
    times = all_depth_times[1][1:nframes]
    years = times ./ SECONDS_PER_YEAR

    @info "Recording depth animation..." variable = var outname nframes framerate
    progress_step = max(1, cld(nframes, 20))
    record(fig, outname, 1:nframes; framerate = framerate) do frame
        fig_title.text = "$(get(VAR_TITLES, var, var)) | Year = $(round(years[frame], digits = 2))"
        for d in 1:nd
            Z[d][] = all_depth_data[d][frame]
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
                                              framerate::Int = 6)
    times, all_data = load_surface_timeseries(files, vars)
    nframes = minimum(length.(values(all_data)))
    nframes > 0 || error("No sea-ice surface frames found for requested variables.")
    if any(length(all_data[v]) != nframes for v in vars)
        @warn "Sea-ice variables have inconsistent frame counts; truncating to shortest." nframes
    end

    idx, xproj, yproj = polar_region_geometry(grid; hemisphere, latitude_cutoff)
    projected_data = Dict{String, Vector{Vector{Float32}}}()

    for var in vars
        projected_data[var] = Vector{Vector{Float32}}(undef, nframes)
        for frame in 1:nframes
            projected_data[var][frame] = vec(all_data[var][frame])[idx]
        end
    end

    observables = Dict{String, Observable{Vector{Float32}}}()
    for var in vars
        observables[var] = Observable(projected_data[var][1])
    end

    hemi_title = hemisphere === :north ? "Arctic" : "Southern Ocean"
    fig = Figure(size = (1400, 900))
    fig_title = Label(fig[0, :], "$(hemi_title) loading...", tellwidth = false)

    axs = Axis[]
    for (panel_idx, var) in enumerate(vars)
        row = cld(panel_idx, 2)
        col = (panel_idx - 1) % 2 + 1
        ax = Axis(fig[row, col], title = "$(get(VAR_TITLES, var, var)) ($(hemi_title))")
        push!(axs, ax)
        hidedecorations!(ax)
        hidespines!(ax)
        cmap, clim = sea_ice_color_settings(var, all_data[var][1])
        hm = scatter!(ax, xproj, yproj;
                      color = observables[var],
                      colormap = cmap,
                      colorrange = clim,
                      marker = :circle,
                      markersize = 3)
        Colorbar(fig[row + 2, col], hm, vertical = false)
    end

    xpad = 0.05f0 * (maximum(xproj) - minimum(xproj))
    ypad = 0.05f0 * (maximum(yproj) - minimum(yproj))
    xlims = (minimum(xproj) - xpad, maximum(xproj) + xpad)
    ylims = (minimum(yproj) - ypad, maximum(yproj) + ypad)

    for ax in axs
        ax.aspect = DataAspect()
        xlims!(ax, xlims)
        ylims!(ax, ylims)
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
            observables[var][] = projected_data[var][frame]
        end
        if frame == 1 || frame == nframes || frame % progress_step == 0
            log_record_progress("sea_ice_$(Symbol(hemisphere))", frame, nframes)
        end
    end

    @info "Saved sea-ice polar animation." hemisphere outname
    return outname
end

function run_all_animations()
    depth_files = depth_slice_files(OUTPUT_PATH)
    isempty(depth_files) && error("No depth-slice files found in $(OUTPUT_PATH).")

    grid_file = replace(first(depth_files), r"\.jld2$" => "")
    grid = create_grid(grid_file; gridtype = "TripolarGrid")

    depths = selected_depth_levels(depth_files)
    runs = unique_iterations(depth_files)
    depths_actual = abs.(grid.z.cᵃᵃᶠ[depths])

    for var in ("T", "S", "u", "v", "speed")
        @info "Processing depth variable..." variable = var
        make_depth_variable_video(var, depths, depths_actual, runs;
                                  outname = FIGDIR * "$(var)_$(RESOLUTION)_all_depths.mp4")
    end

    sea_ice_files = sea_ice_surface_files(OUTPUT_PATH)
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
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_animations()
end
