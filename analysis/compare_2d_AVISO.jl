# Compare monthly model sea-surface height with an AVISO monthly climatology.
#
# Temporal coarsening:
#   Model frames are AveragedTimeInterval values over local-run intervals
#   [t[n-1], t[n]]. The first frame in each run is skipped because its averaging
#   start is unavailable. These interval means are conservatively accumulated
#   into complete no-leap RYF calendar months: intervals crossing a month are
#   split and weighted by overlap duration. Incomplete months are excluded.
#   This matches the coarser AVISOMonthly interval before comparison.
#
# AVISO climatology:
#   The monthly product exposes only SLA. For each calendar month, finite
#   AVISOMonthly :sea_level_anomaly fields are averaged over 1993-2012 by
#   default, then the time-independent MDT is added to reconstruct ADT. MDT is
#   obtained once on the same grid as AVISODaily ADT minus AVISODaily SLA. Raw
#   inputs are fetched in two requests: one monthly SLA time series and one
#   daily ADT+SLA snapshot. The climatology is computed here in
#   build_aviso_climatology_cache and cached under
#   COMPARE_2D_OUTPUT_PATH; raw AVISO downloads use AVISO_DATA_DIR or the
#   NumericalEarth scratch directory.
#
# Spatial coarsening:
#   The finer dataset is interpolated to the coarser grid before subtraction.
#   For the current sxtdeg model, AVISO 1/8 degree is finer than the nominal
#   model 1/6 degree, so AVISO is interpolated to the model tripolar grid. If a
#   future model is finer than 1/8 degree, monthly model means are interpolated
#   to the native AVISO grid instead.

const CURRENT_NUMERICAL_EARTH = expanduser(get(ENV, "NUMERICAL_EARTH_PATH", "/home/tsohail/uom/NumericalEarth-3.jl"))

# Load only side-effect-free comparison utilities.
include(joinpath(@__DIR__, "compare_2d_utils.jl"))

using NumericalEarth

# The mandated submission project currently resolves the older NumericalEarth
# checkout. Load only the compatible AVISO module from NumericalEarth-3, without
# replacing or mutating the active project. The test project exposes the weak
# CopernicusMarine dependency already installed in the shared depot.
insert!(LOAD_PATH, 2, joinpath(CURRENT_NUMERICAL_EARTH, "test"))
using CopernicusMarine
Base.include(NumericalEarth.DataWrangling, joinpath(CURRENT_NUMERICAL_EARTH, "src", "DataWrangling", "AVISO", "AVISO.jl"))
using NumericalEarth.DataWrangling.AVISO: AVISODaily, AVISOMonthly,
                                              copernicusmarine_dataset_id,
                                              copernicusmarine_dataset_version
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: interpolate!
using NCDatasets: Dataset

const AVISO_CLIMATOLOGY_START_YEAR = parse(Int, get(ENV, "AVISO_CLIMATOLOGY_START_YEAR", "1993"))
const AVISO_CLIMATOLOGY_END_YEAR = parse(Int, get(ENV, "AVISO_CLIMATOLOGY_END_YEAR", "2012"))
const AVISO_DATA_DIR = expanduser(get(ENV, "AVISO_DATA_DIR", joinpath(ANALYSIS_OUTPUT_PATH, "AVISO")))
const AVISO_REBUILD_CLIMATOLOGY = lowercase(get(ENV, "COMPARE_2D_AVISO_REBUILD_CLIMATOLOGY", "false")) in ("1", "true", "yes")
const SSH_FILE_PATTERNS = haskey(ENV, "COMPARE_2D_AVISO_FILE_PATTERN") ?
    (ENV["COMPARE_2D_AVISO_FILE_PATTERN"],) :
    ("combined_global_surface_fluxes_$(RESOLUTION)_RYF_run*.jld2",
     "combined_global_ssh_fields_$(RESOLUTION)_RYF_run*.jld2")
const MAX_SSH_INTERVAL = parse(Float64, get(ENV, "COMPARE_2D_AVISO_MAX_INTERVAL_DAYS", "7")) * SECONDS_PER_DAY
const SSH_VARIABLE = "SSH"
const AVISO_SPACING = 1 / 8
const AVISO_DERIVATION = "adt = monthly sla + daily (adt - sla) MDT"
const AVISO_ERROR_COLORRANGE_STD_MULTIPLIER =
    parse(Float32, get(ENV, "COMPARE_2D_AVISO_ERROR_COLORRANGE_STD_MULTIPLIER", "4.0"))

silent_jldopen(f::Function, args...; kwargs...) = with_logger(NullLogger()) do
    JLD2.jldopen(f, args...; kwargs...)
end

copernicus_credential(primary, legacy) = haskey(ENV, primary) ? ENV[primary] : get(ENV, legacy, nothing)
aviso_monthly_batch_filename(years) = "sla_AVISOMonthly_$(first(years))-$(last(years)).nc"

function valid_aviso_batch(path, variables; time_count)
    isfile(path) || return false
    ds = try
        Dataset(path)
    catch
        return false
    end
    valid = try
        all(variable -> haskey(ds, variable), variables) &&
        all(variable -> size(ds[variable], 3) == time_count, variables)
    catch
        false
    finally
        close(ds)
    end
    return valid
end

function download_aviso_batch(dataset, variables, start_date, end_date, output_filename; time_count)
    mkpath(AVISO_DATA_DIR)
    output_path = joinpath(AVISO_DATA_DIR, output_filename)
    valid_aviso_batch(output_path, variables; time_count) && return output_path
    isfile(output_path) && rm(output_path; force = true)

    username = copernicus_credential("COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUS_USERNAME")
    password = copernicus_credential("COPERNICUSMARINE_SERVICE_PASSWORD", "COPERNICUS_PASSWORD")
    kwargs = (; coordinates_selection_method = "outside",
              skip_existing = true,
              dataset_id = copernicusmarine_dataset_id(dataset),
              dataset_version = copernicusmarine_dataset_version(dataset),
              variable = variables,
              start_datetime = string(start_date),
              end_datetime = string(end_date),
              output_filename,
              output_directory = AVISO_DATA_DIR)
    if !isnothing(username) && !isnothing(password)
        kwargs = merge(kwargs, (; username, password))
    end
    CopernicusMarine.subset(; kwargs...)
    valid_aviso_batch(output_path, variables; time_count) ||
        error("Copernicus Marine produced an invalid AVISO batch file: $output_path")
    return output_path
end

function load_aviso_cached_frame(filepath::AbstractString, var::String, index::Int)
    return silent_jldopen(filepath, "r") do file
        extract_2d_f32(file[string(var, "/", index)])
    end
end

function aviso_cached_frame_reader(filepath::AbstractString)
    handle = Ref{Any}(nothing)
    ensure_handle() = handle[] === nothing ? (handle[] = with_logger(NullLogger()) do; JLD2.jldopen(filepath, "r"); end) : handle[]
    function load_frame!(dest::Matrix{Float32}, var::String, index::Int)
        return with_logger(NullLogger()) do
            copy_2d_to!(dest, ensure_handle()[string(var, "/", index)])
        end
    end
    function close_reader!()
        handle[] !== nothing && close(handle[])
        handle[] = nothing
        return nothing
    end
    return (; load_frame!, close_reader!)
end

function ssh_files(path::AbstractString = OUTPUT_PATH; patterns = SSH_FILE_PATTERNS)
    files = String[]
    for pattern in patterns
        append!(files, glob(pattern, path))
    end
    unique!(files)
    filter!(file -> !occursin("_rank", basename(file)) && run_number(file) >= 0, files)
    # The dedicated SSH output is the valid source before SSH was moved into
    # surface-flux output. Process it last so it wins equal-run timestamp ties.
    sort!(files; by = file -> (run_number(file), occursin("_ssh_fields_", basename(file)) ? 1 : 0, basename(file)))
    return files
end

function ssh_keys(file)
    haskey(file, "timeseries/t") || return Int[]
    haskey(file, "timeseries/surface_height") || return Int[]
    tkeys = Set(numeric_timeseries_keys(file["timeseries/t"]))
    hkeys = Set(numeric_timeseries_keys(file["timeseries/surface_height"]))
    return sort!(collect(intersect(tkeys, hkeys)))
end

function index_ssh_intervals(; path::AbstractString = OUTPUT_PATH, patterns = SSH_FILE_PATTERNS)
    files = ssh_files(path; patterns)
    isempty(files) && error("No SSH files matching $(join(patterns, ", ")) in $(path).")
    records_by_end = Dict{Float64, NamedTuple}()
    grid = nothing
    skipped_first = 0
    replaced_duplicates = 0
    skipped_gaps = 0

    for (file_index, filepath) in enumerate(files)
        run = run_number(filepath)
        grid === nothing && (grid = load_grid_from_output_file(filepath))
        silent_jldopen(filepath, "r") do file
            keys = ssh_keys(file)
            isempty(keys) && return
            skipped_first += 1
            for n in 2:length(keys)
                key = keys[n]
                previous_key = keys[n - 1]
                interval_start = Float64(file["timeseries/t/$previous_key"])
                interval_end = Float64(file["timeseries/t/$key"])
                interval_end > interval_start || continue
                if interval_end - interval_start > MAX_SSH_INTERVAL
                    skipped_gaps += 1
                    continue
                end
                existing = get(records_by_end, interval_end, nothing)
                if existing === nothing || run >= existing.run
                    replaced_duplicates += existing === nothing ? 0 : 1
                    records_by_end[interval_end] = (; run, filepath, key, interval_start, interval_end)
                end
            end
        end
        if file_index == 1 || file_index == length(files) || file_index % max(1, cld(length(files), PROGRESS_UPDATES)) == 0
            log_progress("index_SSH", file_index, length(files))
        end
    end

    records = [records_by_end[t] for t in sort!(collect(keys(records_by_end)))]
    isempty(records) && error("No SSH averaging intervals were found.")
    grid === nothing && error("No serialized model grid was found.")
    @info "Indexed interval-averaged SSH." files = length(files) intervals = length(records) skipped_first replaced_duplicates skipped_gaps
    return (; records, grid)
end

load_ssh_frame(record) = silent_jldopen(record.filepath, "r") do file
    extract_2d_f32(file["timeseries/surface_height/$(record.key)"])
end

function aviso_source_template()
    grid = LatitudeLongitudeGrid(CPU(), Float32;
                                 size = (2880, 1440),
                                 longitude = (-180, 180),
                                 latitude = (-90, 90),
                                 halo = (3, 3),
                                 topology = (Periodic, Bounded, Flat))
    return Field{Center, Center, Nothing}(grid)
end

function aviso_mean_dynamic_topography(path)
    ds = Dataset(path)
    adt_data = Array(ds["adt"][:, :, 1])
    sla_data = Array(ds["sla"][:, :, 1])
    close(ds)
    mdt = Matrix{Float32}(undef, size(adt_data))
    @inbounds for i in eachindex(mdt, adt_data, sla_data)
        a = adt_data[i]
        s = sla_data[i]
        mdt[i] = !ismissing(a) && !ismissing(s) && isfinite(a) && isfinite(s) ? Float32(a - s) : NaN32
    end
    return mdt
end

function add_mean_dynamic_topography!(adt, sla, mdt)
    @inbounds for i in eachindex(adt, sla, mdt)
        s = Float32(sla[i])
        m = Float32(mdt[i])
        adt[i] = isfinite(s) && isfinite(m) ? s + m : NaN32
    end
    return adt
end

function comparison_target(model_grid)
    model = underlying_grid(model_grid)
    model_spacing = 360 / model.Nx
    target = model_spacing >= AVISO_SPACING ? :model : :aviso
    @info "Selected coarser spatial comparison grid." target model_spacing_degrees = model_spacing aviso_spacing_degrees = AVISO_SPACING
    return target
end

function comparison_shape(target, model_grid)
    target == :model && return (underlying_grid(model_grid).Nx, underlying_grid(model_grid).Ny)
    return (2880, 1440)
end

function accumulate_finite!(sum_field, count_field, field)
    @inbounds for i in eachindex(sum_field, count_field, field)
        raw_value = field[i]
        if !ismissing(raw_value)
            value = Float64(raw_value)
            if isfinite(value)
                sum_field[i] += value
                count_field[i] += 1
            end
        end
    end
    return nothing
end

function accumulate_weighted!(sum_field, weights, field, dt)
    @inbounds for i in eachindex(sum_field, weights, field)
        value = Float64(field[i])
        if isfinite(value)
            sum_field[i] += value * dt
            weights[i] += dt
        end
    end
    return nothing
end

function finite_mean(sum_field, weights)
    result = Matrix{Float32}(undef, size(sum_field))
    @inbounds for i in eachindex(result, sum_field, weights)
        result[i] = weights[i] > 0 ? Float32(sum_field[i] / weights[i]) : NaN32
    end
    return result
end

function aviso_climatology_cache_path(target, model_grid)
    nx, ny = comparison_shape(target, model_grid)
    filename = "compare_2d_AVISO_climatology_$(AVISO_CLIMATOLOGY_START_YEAR)_$(AVISO_CLIMATOLOGY_END_YEAR)_$(target)_$(RESOLUTION)_$(nx)x$(ny).jld2"
    return joinpath(ANALYSIS_OUTPUT_PATH, filename)
end

function valid_aviso_cache(path, target, model_grid)
    isfile(path) || return false
    expected = comparison_shape(target, model_grid)
    return try
        silent_jldopen(path, "r") do file
            haskey(file, "derived_variable") &&
            file["derived_variable"] == AVISO_DERIVATION &&
            all(month -> haskey(file, "$SSH_VARIABLE/$month") && size(extract_2d(file["$SSH_VARIABLE/$month"])) == expected, 1:12)
        end
    catch
        false
    end
end

function build_aviso_climatology_cache(model_grid, target; rebuild = AVISO_REBUILD_CLIMATOLOGY)
    AVISO_CLIMATOLOGY_END_YEAR >= AVISO_CLIMATOLOGY_START_YEAR || error("Invalid AVISO climatology range.")
    mkpath(ANALYSIS_OUTPUT_PATH)
    cache_path = aviso_climatology_cache_path(target, model_grid)
    if !rebuild && valid_aviso_cache(cache_path, target, model_grid)
        @info "Reusing spatially coarsened AVISO climatology." cache_path target
        return cache_path
    end

    interpolation_target = target == :model ? Field{Center, Center, Nothing}(materialized_underlying_grid(model_grid)) : nothing
    years = AVISO_CLIMATOLOGY_START_YEAR:AVISO_CLIMATOLOGY_END_YEAR
    monthly_start = DateTime(first(years), 1, 1)
    monthly_end = DateTime(last(years), 12, 1)
    monthly_count = length(years) * 12
    monthly_path = download_aviso_batch(AVISOMonthly(), ["sla"], monthly_start, monthly_end,
                                        aviso_monthly_batch_filename(years);
                                        time_count = monthly_count)
    mdt_date = DateTime(first(years), 1, 1)
    daily_path = download_aviso_batch(AVISODaily(), ["adt", "sla"], mdt_date, mdt_date,
                                      "adt_sla_AVISODaily_$(Dates.format(mdt_date, "yyyy-mm-dd")).nc";
                                      time_count = 1)
    mean_dynamic_topography = aviso_mean_dynamic_topography(daily_path)
    monthly_ds = Dataset(monthly_path)
    source_template = aviso_source_template()

    silent_jldopen(cache_path, "w") do cache
        cache["climatology_start_year"] = AVISO_CLIMATOLOGY_START_YEAR
        cache["climatology_end_year"] = AVISO_CLIMATOLOGY_END_YEAR
        cache["source_variable"] = "sla"
        cache["derived_variable"] = AVISO_DERIVATION
        cache["comparison_grid"] = String(target)

        for month in 1:12
            sum_field = nothing
            count_field = nothing
            for (year_index, year) in enumerate(years)
                time_index = 12 * (year - first(years)) + month
                source_data = Array(monthly_ds["sla"][:, :, time_index])
                if sum_field === nothing
                    sum_field = zeros(Float64, size(source_data))
                    count_field = zeros(Int32, size(source_data))
                end
                accumulate_finite!(sum_field, count_field, source_data)
                if year_index == 1 || year_index == length(years) || year_index % max(1, cld(length(years), PROGRESS_UPDATES)) == 0
                    log_progress("AVISO_month_$(month)", year_index, length(years))
                end
            end

            monthly_sla_climatology = finite_mean(sum_field, count_field)
            native_climatology = similar(monthly_sla_climatology)
            add_mean_dynamic_topography!(native_climatology, monthly_sla_climatology, mean_dynamic_topography)
            if target == :model
                interior(source_template, :, :, 1) .= native_climatology
                fill_halo_regions!(source_template)
                interpolate!(interpolation_target, source_template)
                cache["$SSH_VARIABLE/$month"] = extract_2d_f32(Array(interior(interpolation_target)))
            else
                cache["$SSH_VARIABLE/$month"] = native_climatology
            end
            @info "Cached AVISO climatology month." month target cache_path
        end
    end
    close(monthly_ds)
    return cache_path
end

function write_model_month!(cache, frame, monthly_mean, target, model_source, aviso_target)
    if target == :model
        cache["$SSH_VARIABLE/$frame"] = monthly_mean
    else
        interior(model_source, :, :, 1) .= monthly_mean
        fill_halo_regions!(model_source)
        interpolate!(aviso_target, model_source)
        cache["$SSH_VARIABLE/$frame"] = extract_2d_f32(Array(interior(aviso_target)))
    end
    return nothing
end

function build_monthly_model_cache(index, target, cache_path)
    model_source = target == :aviso ? Field{Center, Center, Nothing}(materialized_underlying_grid(index.grid)) : nothing
    aviso_target = if target == :aviso
        aviso_source_template()
    else
        nothing
    end
    bins = Int[]
    years = Int[]
    months = Int[]
    midpoints = Float64[]
    frame = 0
    current_bin = nothing
    current_year = current_month = 0
    weighted_sum = nothing
    finite_weights = nothing
    coverage = 0.0

    silent_jldopen(cache_path, "w") do cache
        function flush_current!()
            current_bin === nothing && return
            expected = NOLEAP_MONTH_DAYS[current_month] * SECONDS_PER_DAY
            tolerance = max(1.0, expected * 1e-8)
            if abs(coverage - expected) <= tolerance
                frame += 1
                monthly_mean = finite_mean(weighted_sum, finite_weights)
                write_model_month!(cache, frame, monthly_mean, target, model_source, aviso_target)
                push!(bins, current_bin)
                push!(years, current_year)
                push!(months, current_month)
                month_start = current_year * SECONDS_PER_YEAR + CUMULATIVE_MONTH_SECONDS[current_month]
                push!(midpoints, month_start + expected / 2)
            else
                @warn "Skipping incomplete model month." year_index = current_year month = current_month coverage_days = coverage / SECONDS_PER_DAY expected_days = expected / SECONDS_PER_DAY
            end
            return nothing
        end

        for (record_index, record) in enumerate(index.records)
            field = load_ssh_frame(record)
            for (year_index, month, piece_start, piece_end) in split_interval_by_month(record.interval_start, record.interval_end)
                bin = 12 * year_index + month
                if current_bin != bin
                    flush_current!()
                    current_bin = bin
                    current_year = year_index
                    current_month = month
                    weighted_sum = zeros(Float64, size(field))
                    finite_weights = zeros(Float64, size(field))
                    coverage = 0.0
                end
                dt = piece_end - piece_start
                accumulate_weighted!(weighted_sum, finite_weights, field, dt)
                coverage += dt
            end
            if record_index == 1 || record_index == length(index.records) || record_index % max(1, cld(length(index.records), PROGRESS_UPDATES)) == 0
                log_progress("monthly_model_SSH", record_index, length(index.records))
            end
        end
        flush_current!()
        cache["bins"] = bins
        cache["years"] = years
        cache["months"] = months
        cache["midpoints"] = midpoints
        cache["comparison_grid"] = String(target)
    end

    isempty(bins) && error("No complete model months were covered by the SSH intervals.")
    @info "Built complete monthly model SSH cache." months = length(bins) target cache_path
    return (; cache_path, bins, years, months, midpoints)
end

function comparison_mask(target, model_grid, aviso_cache)
    if target == :model
        bottom = bottom_height_matrix(model_grid)
        return isnothing(bottom) ? nothing : bottom .< 0f0
    end

    mask = falses(comparison_shape(target, model_grid))
    for month in 1:12
        mask .|= isfinite.(load_aviso_cached_frame(aviso_cache, SSH_VARIABLE, month))
    end
    return mask
end

function comparison_colorranges(model_cache, aviso_cache, aviso_months, mask)
    frame = lastindex(aviso_months)
    model = load_aviso_cached_frame(model_cache, SSH_VARIABLE, frame)
    aviso = load_aviso_cached_frame(aviso_cache, SSH_VARIABLE, aviso_months[frame])
    model0 = load_aviso_cached_frame(model_cache, SSH_VARIABLE, 1)
    aviso0 = load_aviso_cached_frame(aviso_cache, SSH_VARIABLE, aviso_months[1])

    nvalue = nerror = 0
    mean_value = mean_error = 0.0
    m2_value = m2_error = 0.0

    for (field, reference) in ((model, model0), (aviso, aviso0))
        @inbounds for i in eachindex(field, reference)
            if (isnothing(mask) || mask[i]) && isfinite(field[i]) && isfinite(reference[i])
                value = Float64(field[i] - reference[i])
                nvalue += 1
                delta = value - mean_value
                mean_value += delta / nvalue
                m2_value += delta * (value - mean_value)
            end
        end
    end

    @inbounds for i in eachindex(model, aviso, model0, aviso0)
        if (isnothing(mask) || mask[i]) && isfinite(model[i]) && isfinite(aviso[i]) && isfinite(model0[i]) && isfinite(aviso0[i])
            value = Float64((model[i] - model0[i]) - (aviso[i] - aviso0[i]))
            nerror += 1
            delta = value - mean_error
            mean_error += delta / nerror
            m2_error += delta * (value - mean_error)
        end
    end

    nvalue > 1 || return ((-1f0, 1f0), (-1f0, 1f0))
    value_halfwidth = max(Float32(VALUE_COLORRANGE_STD_MULTIPLIER * sqrt(m2_value / (nvalue - 1))), 1f-6)
    value_limit = abs(Float32(mean_value)) + value_halfwidth
    error_halfwidth = nerror > 1 ? max(Float32(AVISO_ERROR_COLORRANGE_STD_MULTIPLIER * sqrt(m2_error / (nerror - 1))), 1f-6) : 1f0
    @info "Set symmetric anomaly color ranges from final monthly timestep." frame month = aviso_months[frame] error_std_multiplier = AVISO_ERROR_COLORRANGE_STD_MULTIPLIER
    return ((-value_limit, value_limit), (-error_halfwidth, error_halfwidth))
end

function make_aviso_video(metadata, aviso_cache, mask; outname)
    nframes = length(metadata.bins)
    aviso_months = mod1.(metadata.bins, 12)
    model0 = load_aviso_cached_frame(metadata.cache_path, SSH_VARIABLE, 1)
    aviso0 = load_aviso_cached_frame(aviso_cache, SSH_VARIABLE, aviso_months[1])
    model_obs = Observable(build_error_frame(model0, model0, mask))
    aviso_obs = Observable(build_error_frame(aviso0, aviso0, mask))
    error_obs = Observable(build_error_frame(model_obs[], aviso_obs[], mask))
    model_buffer = similar(model0)
    aviso_buffer = similar(aviso0)
    model_reader = aviso_cached_frame_reader(metadata.cache_path)
    aviso_reader = aviso_cached_frame_reader(aviso_cache)
    value_range, error_range = comparison_colorranges(metadata.cache_path, aviso_cache, aviso_months, mask)

    fig = Figure(size = (760, 1080))
    title = Label(fig[0, 1], "", tellwidth = false)
    ax_model = Axis(fig[1, 1], title = "Model monthly-mean SSH change from first month", ylabel = "Model")
    hm_model = heatmap!(ax_model, model_obs; colormap = :balance, colorrange = value_range, nan_color = NAN_PLOT_COLOR)
    ax_aviso = Axis(fig[2, 1], title = "AVISO ADT climatology change from first month", ylabel = "AVISO")
    heatmap!(ax_aviso, aviso_obs; colormap = :balance, colorrange = value_range, nan_color = NAN_PLOT_COLOR)
    ax_error = Axis(fig[3, 1], title = "Model - AVISO SSH change", ylabel = "Difference")
    hm_error = heatmap!(ax_error, error_obs; colormap = :balance, colorrange = error_range, nan_color = NAN_PLOT_COLOR)
    Colorbar(fig[4, 1], hm_model; vertical = false, label = "SSH change from first month (m)")
    Colorbar(fig[5, 1], hm_error; vertical = false, label = "SSH-change difference (m)")
    resize_to_layout!(fig)

    framerate = constant_model_dt_framerate(Float64.(metadata.midpoints))
    progress_step = max(1, cld(nframes, PROGRESS_UPDATES))
    try
        record_video_in_chunks(fig, outname, 1:nframes, framerate; tag = "compare_2d_AVISO") do frame
            month = aviso_months[frame]
            model_reader.load_frame!(model_buffer, SSH_VARIABLE, frame) || error("Could not load model month $(frame).")
            aviso_reader.load_frame!(aviso_buffer, SSH_VARIABLE, month) || error("Could not load AVISO month $(month).")
            build_error_frame!(model_obs[], model_buffer, model0, mask)
            build_error_frame!(aviso_obs[], aviso_buffer, aviso0, mask)
            build_error_frame!(error_obs[], model_obs[], aviso_obs[], mask)
            notify(model_obs); notify(aviso_obs); notify(error_obs)
            title.text = "SSH / AVISO climatology | RYF year $(metadata.years[frame] + 1) | $(Dates.format(Date(2001, month, 1), "mmmm"))"
            frame % GC_INTERVAL == 0 && GC.gc(false)
            if frame == 1 || frame == nframes || frame % progress_step == 0
                log_progress("video_compare_2d_AVISO", frame, nframes)
            end
        end
    finally
        model_reader.close_reader!()
        aviso_reader.close_reader!()
    end

    @info "Saved monthly AVISO SSH comparison animation." outname nframes framerate value_range error_range
    return outname
end

function run_AVISO_comparison()
    mkpath(FIGDIR)
    mkpath(ANALYSIS_OUTPUT_PATH)
    scratch_dir = mktempdir(ANALYSIS_OUTPUT_PATH; prefix = "compare_2d_AVISO_scratch_")
    try
        index = index_ssh_intervals()
        target = comparison_target(index.grid)
        aviso_cache = build_aviso_climatology_cache(index.grid, target)
        model_cache = build_monthly_model_cache(index, target, joinpath(scratch_dir, "monthly_model_SSH.jld2"))
        mask = comparison_mask(target, index.grid, aviso_cache)
        video_file = joinpath(FIGDIR, "compare_2d_AVISO_SSH_$(RESOLUTION).mp4")
        make_aviso_video(model_cache, aviso_cache, mask; outname = video_file)
        @info "Finished 2D AVISO comparison." video_file aviso_cache months = length(model_cache.months) target
        return video_file
    finally
        rm(scratch_dir; recursive = true, force = true)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_AVISO_comparison()
end
