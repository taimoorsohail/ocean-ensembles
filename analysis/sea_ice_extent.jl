using CairoMakie
using JLD2
using Oceananigans
using Dates
using Downloads
using Glob
using Logging
using Statistics

with_trailing_slash(path) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator

const OUTPUT_PATH = with_trailing_slash(expanduser(get(ENV, "OUTPUT_PATH", "/data/gpfs/projects/punim2499/taimoor/ocean-ensembles/outputs/saved/")))
const FIGDIR = with_trailing_slash(expanduser(get(ENV, "FIGDIR", "/data/gpfs/projects/punim2499/taimoor/ocean-ensembles/figures/")))
const NSIDC_DATA_DIR = with_trailing_slash(expanduser(get(ENV, "NSIDC_DATA_DIR", "/home/tsohail/uom/ocean-ensembles/data/nsidc/")))
const NSIDC_ARCTIC_URL = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/data/N_seaice_extent_daily_v4.0.csv"
const NSIDC_ANTARCTIC_URL = "https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/data/S_seaice_extent_daily_v4.0.csv"
const NSIDC_ARCTIC_FILE = expanduser(get(ENV, "NSIDC_ARCTIC_FILE", NSIDC_DATA_DIR * "N_seaice_extent_daily_v4.0.csv"))
const NSIDC_ANTARCTIC_FILE = expanduser(get(ENV, "NSIDC_ANTARCTIC_FILE", NSIDC_DATA_DIR * "S_seaice_extent_daily_v4.0.csv"))
const NSIDC_CLIM_START_YEAR = parse(Int, get(ENV, "NSIDC_CLIM_START_YEAR", "1981"))
const NSIDC_CLIM_END_YEAR = parse(Int, get(ENV, "NSIDC_CLIM_END_YEAR", "2010"))
const RESOLUTION = "sxtdeg"
const ICE_THRESHOLD = 0.15f0
const SECONDS_PER_DAY = 24 * 60 * 60
const RYF_YEAR_DAYS = 365.0
const PROGRESS_UPDATES = 20
const SURFACE_PATTERN = "combined_global_surface_fluxes_$(RESOLUTION)_RYF_run*.jld2"

function print_usage()
    println("Usage: julia --project=ocean-ensembles/experiments/submission_scripts ocean-ensembles/analysis/sea_ice_extent.jl")
    println("Builds Northern and Southern Hemisphere sea-ice extent timeseries from the combined surface-flux run files.")
    println("Overlays the NSIDC Sea Ice Index 1981-2010 daily median extent climatology plus a 10th-90th percentile band.")
    return nothing
end

run_number(file::AbstractString) = begin
    match_result = match(r"_run(\d+)\.jld2$", basename(file))
    isnothing(match_result) ? -1 : parse(Int, match_result.captures[1])
end

function surface_files()
    files = glob(SURFACE_PATTERN, OUTPUT_PATH)
    filter!(file -> run_number(file) >= 0, files)
    sort!(files; by = run_number)
    isempty(files) && error("No files matching $SURFACE_PATTERN were found in $OUTPUT_PATH.")
    return files
end

@inline function extract_2d_f32(raw)
    if ndims(raw) == 2
        return Float32.(raw)
    elseif ndims(raw) == 3
        return Float32.(raw[:, :, 1])
    end
    return nothing
end

function load_grid_from_output_file(filepath::AbstractString, variable::AbstractString)
    return with_logger(NullLogger()) do
        jldopen(filepath, "r") do file
            haskey(file, "serialized/grid") && return file["serialized/grid"]

            grid_index_path = "timeseries/$variable/serialized/grid_index"
            haskey(file, grid_index_path) || error("No serialized grid found for $variable in $filepath.")
            grid_path = "serialized/grid_$(Int(file[grid_index_path]))"
            haskey(file, grid_path) || error("Missing $grid_path referenced by $variable in $filepath.")
            return file[grid_path]
        end
    end
end

underlying_grid(grid) = hasproperty(grid, :underlying_grid) ? getproperty(grid, :underlying_grid) : grid

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

parent_array(source) = hasproperty(source, :parent) ? getproperty(source, :parent) : Array(source)

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
    source_grid = underlying_grid(grid)
    immersed_boundary = getproperty(grid, :immersed_boundary)
    hasproperty(immersed_boundary, :bottom_height) || return nothing
    bottom_height_field = getproperty(immersed_boundary, :bottom_height)
    hasproperty(bottom_height_field, :data) || return nothing
    return physical_matrix(getproperty(bottom_height_field, :data), source_grid; T = Float32)
end

surface_ocean_mask(bottom_height::Union{Nothing, AbstractMatrix}) = isnothing(bottom_height) ? nothing : BitMatrix(bottom_height .< 0f0)

function center_latitude(grid)
    g = underlying_grid(grid)
    hasproperty(g, :φᶜᶜᵃ) || error("Serialized grid does not contain center latitude nodes.")
    return physical_matrix(getproperty(g, :φᶜᶜᵃ), g; T = Float32)
end

function cell_area_matrix(grid)
    g = underlying_grid(grid)
    hasproperty(g, :Azᶜᶜᵃ) || error("Serialized grid does not contain center cell areas.")
    return physical_matrix(getproperty(g, :Azᶜᶜᵃ), g; T = Float64)
end

function hemisphere_masks(grid, wet_mask::Union{Nothing, BitMatrix})
    lat = center_latitude(grid)
    north_mask = lat .> 0f0
    south_mask = lat .< 0f0

    if !isnothing(wet_mask)
        north_mask .&= wet_mask
        south_mask .&= wet_mask
    end

    any(north_mask) || error("No wet cells found in the Northern Hemisphere.")
    any(south_mask) || error("No wet cells found in the Southern Hemisphere.")
    return Dict(:north => BitMatrix(north_mask), :south => BitMatrix(south_mask))
end

function sea_ice_extent(ice::AbstractMatrix, areas::AbstractMatrix, mask::BitMatrix; threshold::Float32 = ICE_THRESHOLD)
    size(ice) == size(areas) || error("Area shape mismatch: got $(size(areas)) expected $(size(ice)).")
    size(mask) == size(ice) || error("Mask shape mismatch: got $(size(mask)) expected $(size(ice)).")

    total = 0.0
    @inbounds for idx in eachindex(ice, areas, mask)
        mask[idx] || continue
        concentration = ice[idx]
        if isfinite(concentration) && concentration > threshold
            total += areas[idx]
        end
    end
    return total
end

function load_sea_ice_extent_timeseries(files::Vector{String}; threshold::Float32 = ICE_THRESHOLD)
    grid = load_grid_from_output_file(last(files), "ice_concentration")
    areas = cell_area_matrix(grid)
    wet_mask = surface_ocean_mask(bottom_height_matrix(grid))
    masks = hemisphere_masks(grid, wet_mask)

    records_by_time = Dict{Float64, Tuple{Float64, Float64}}()

    for file in files
        @info "Loading sea-ice extent timeseries" file = basename(file) threshold
        jldopen(file, "r") do data
            haskey(data, "timeseries/t") || error("Missing timeseries/t in $file")
            haskey(data, "timeseries/ice_concentration") || error("Missing timeseries/ice_concentration in $file")

            ts_keys = sort(parse.(Int, collect(keys(data["timeseries/t"]))))
            stride = max(1, cld(length(ts_keys), PROGRESS_UPDATES))

            for (index, key) in enumerate(ts_keys)
                raw = data["timeseries/ice_concentration/$key"]
                ice = extract_2d_f32(raw)
                ice === nothing && continue

                time = Float64(data["timeseries/t/$key"])
                records_by_time[time] = (sea_ice_extent(ice, areas, masks[:north]; threshold),
                                         sea_ice_extent(ice, areas, masks[:south]; threshold))

                if index == 1 || index == length(ts_keys) || index % stride == 0
                    pct = round(100 * index / length(ts_keys); digits = 1)
                    @info "Sea-ice extent scan" progress = "$(index)/$(length(ts_keys))" percent = pct
                end
            end
        end
    end

    times = sort!(collect(keys(records_by_time)))
    isempty(times) && error("No sea-ice extent frames found in $(join(basename.(files), ", ")).")
    north_extents = [records_by_time[time][1] for time in times]
    south_extents = [records_by_time[time][2] for time in times]
    @info "Completed sea-ice extent load." frames = length(times)
    return times, north_extents, south_extents
end

ryf_dates() = vcat(collect(DateTime(1991, 1, 1):Month(1):DateTime(1991, 4, 1)),
                   collect(DateTime(1990, 5, 1):Month(1):DateTime(1990, 12, 1)))

function ryf_month_starts_days()
    dates = ryf_dates()
    reference_date = dates[1]
    starts = Float64[Dates.value(date - reference_date) / (1000 * 60 * 60 * 24) for date in dates]
    return starts
end

function positive_extent_xlim(times_days::Vector{Float64}, north_extents::Vector{Float64}, south_extents::Vector{Float64})
    positive = findall(i -> north_extents[i] > 0 || south_extents[i] > 0, eachindex(times_days))
    isempty(positive) && return nothing
    return (times_days[first(positive)], times_days[last(positive)])
end

function ryf_month_ticks(xlimits::Tuple{Float64, Float64})
    month_starts = ryf_month_starts_days()
    month_labels = Dates.format.(ryf_dates(), "u")

    xmin, xmax = xlimits
    start_year = floor(Int, xmin / RYF_YEAR_DAYS)
    end_year = ceil(Int, xmax / RYF_YEAR_DAYS)

    positions = Float64[]
    labels = String[]
    for year in start_year:end_year
        offset = year * RYF_YEAR_DAYS
        for (month_start, month_label) in zip(month_starts, month_labels)
            position = offset + month_start
            if xmin <= position <= xmax
                push!(positions, position)
                push!(labels, month_label)
            end
        end
    end

    return positions, labels
end

function ensure_nsidc_file(local_path::AbstractString, url::AbstractString)
    isfile(local_path) && return local_path
    mkpath(dirname(local_path))
    @info "Downloading NSIDC daily extent file..." local_path url
    try
        Downloads.download(url, local_path)
    catch err
        error("Could not download NSIDC daily extent file from $(url). Set the local file explicitly via environment variables if needed. Original error: $(sprint(showerror, err))")
    end
    return local_path
end

function parse_nsidc_daily_extent(local_path::AbstractString)
    by_doy = [Float64[] for _ in 1:Int(RYF_YEAR_DAYS)]

    open(local_path, "r") do io
        for line in eachline(io)
            stripped = strip(line)
            isempty(stripped) && continue
            startswith(stripped, "Year") && continue
            startswith(stripped, "YYYY") && continue

            fields = split(line, ',')
            length(fields) >= 4 || continue

            year = tryparse(Int, strip(fields[1]))
            month = tryparse(Int, strip(fields[2]))
            day = tryparse(Int, strip(fields[3]))
            extent = tryparse(Float64, strip(fields[4]))
            any(isnothing, (year, month, day, extent)) && continue
            NSIDC_CLIM_START_YEAR <= year <= NSIDC_CLIM_END_YEAR || continue
            month == 2 && day == 29 && continue

            doy = dayofyear(Date(2001, month, day))
            push!(by_doy[doy], extent)
        end
    end

    all(!isempty, by_doy) || error("NSIDC daily file $(local_path) did not provide enough data to build a full 365-day climatology for $(NSIDC_CLIM_START_YEAR)-$(NSIDC_CLIM_END_YEAR).")
    median = [quantile(values, 0.50) for values in by_doy]
    lower = [quantile(values, 0.10) for values in by_doy]
    upper = [quantile(values, 0.90) for values in by_doy]
    return (; median, lower, upper)
end

function load_nsidc_climatologies()
    north_file = ensure_nsidc_file(NSIDC_ARCTIC_FILE, NSIDC_ARCTIC_URL)
    south_file = ensure_nsidc_file(NSIDC_ANTARCTIC_FILE, NSIDC_ANTARCTIC_URL)
    north = parse_nsidc_daily_extent(north_file)
    south = parse_nsidc_daily_extent(south_file)
    return north, south
end

function tiled_daily_climatology(times_days::Vector{Float64}, climatology::Vector{Float64})
    length(climatology) == Int(RYF_YEAR_DAYS) || error("Expected a 365-day climatology, got $(length(climatology)) days.")

    tiled = Vector{Float64}(undef, length(times_days))
    @inbounds for i in eachindex(times_days)
        day_in_year = mod(times_days[i], RYF_YEAR_DAYS)
        lower_day = floor(Int, day_in_year) + 1
        frac = day_in_year - floor(day_in_year)
        upper_day = lower_day == Int(RYF_YEAR_DAYS) ? 1 : lower_day + 1
        tiled[i] = (1 - frac) * climatology[lower_day] + frac * climatology[upper_day]
    end
    return tiled
end

function plot_sea_ice_extent(times::Vector{Float64}, north_extents::Vector{Float64}, south_extents::Vector{Float64}; outname::AbstractString)
    fig = Figure(size = (1600, 900))

    ax_north = Axis(fig[1, 1],
                    xlabel = "Month",
                    ylabel = "Extent (million km^2)",
                    title = "Northern Hemisphere Sea-Ice Extent")
    ax_south = Axis(fig[2, 1],
                    xlabel = "Month",
                    ylabel = "Extent (million km^2)",
                    title = "Southern Hemisphere Sea-Ice Extent")

    times_days = times ./ SECONDS_PER_DAY
    north_nsidc, south_nsidc = load_nsidc_climatologies()
    north_nsidc_median = tiled_daily_climatology(times_days, north_nsidc.median)
    north_nsidc_lower = tiled_daily_climatology(times_days, north_nsidc.lower)
    north_nsidc_upper = tiled_daily_climatology(times_days, north_nsidc.upper)
    south_nsidc_median = tiled_daily_climatology(times_days, south_nsidc.median)
    south_nsidc_lower = tiled_daily_climatology(times_days, south_nsidc.lower)
    south_nsidc_upper = tiled_daily_climatology(times_days, south_nsidc.upper)

    north_band = band!(ax_north, times_days, north_nsidc_lower, north_nsidc_upper, color = (:black, 0.12))
    south_band = band!(ax_south, times_days, south_nsidc_lower, south_nsidc_upper, color = (:black, 0.12))
    north_model_line = lines!(ax_north, times_days, north_extents ./ 1e12, color = :steelblue4, linewidth = 3, label = "Model")
    south_model_line = lines!(ax_south, times_days, south_extents ./ 1e12, color = :firebrick3, linewidth = 3, label = "Model")
    north_nsidc_line = lines!(ax_north, times_days, north_nsidc_median, color = :black, linewidth = 2.5, linestyle = :dash, label = "NSIDC 1981-2010 median")
    south_nsidc_line = lines!(ax_south, times_days, south_nsidc_median, color = :black, linewidth = 2.5, linestyle = :dash, label = "NSIDC 1981-2010 median")

    xlimits = positive_extent_xlim(times_days, north_extents, south_extents)
    if !isnothing(xlimits)
        xlims!(ax_north, xlimits)
        xlims!(ax_south, xlimits)
        month_positions, month_labels = ryf_month_ticks(xlimits)
        ax_north.xticks = (month_positions, month_labels)
        ax_south.xticks = (month_positions, month_labels)
    end

    axislegend(ax_north,
               [north_model_line, north_nsidc_line, north_band],
               ["Model", "NSIDC 1981-2010 median", "NSIDC 10-90%"],
               position = :rb)
    axislegend(ax_south,
               [south_model_line, south_nsidc_line, south_band],
               ["Model", "NSIDC 1981-2010 median", "NSIDC 10-90%"],
               position = :rb)

    linkxaxes!(ax_north, ax_south)
    rowgap!(fig.layout, 16)
    save(outname, fig)
    return outname
end

function make_sea_ice_extent(; threshold::Float32 = ICE_THRESHOLD,
                              outname::Union{Nothing, String} = nothing)
    times, north_extents, south_extents = load_sea_ice_extent_timeseries(surface_files(); threshold)

    if isnothing(outname)
        outname = FIGDIR * "sea_ice_extent.png"
    end

    plot_sea_ice_extent(times, north_extents, south_extents; outname)
    @info "Saved sea-ice extent plot." outname
    return times, north_extents, south_extents, outname
end

if abspath(PROGRAM_FILE) == @__FILE__
    if any(arg -> arg in ("-h", "--help"), ARGS)
        print_usage()
    else
        isempty(ARGS) || error("This script does not accept positional arguments.")
        make_sea_ice_extent()
    end
end
