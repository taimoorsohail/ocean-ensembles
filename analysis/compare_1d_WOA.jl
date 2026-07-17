using CairoMakie
using NumericalEarth
using ConservativeRegridding
using Dates
using JLD2
using Logging
using OceanEnsembles
using Oceananigans
using Printf
using Statistics
using WorldOceanAtlasTools
using Glob

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

const OUTPUT_PATH = expanduser(get(ENV, "OUTPUT_PATH", "/home/tsohail/uom/ocean-ensembles/outputs/"))
const ANALYSIS_OUTPUT_PATH = expanduser(get(ENV, "COMPARE_1D_OUTPUT_PATH", OUTPUT_PATH))
const FIGDIR = expanduser(get(ENV, "FIGDIR", "/home/tsohail/uom/ocean-ensembles/figures/"))
const RESOLUTION = get(ENV, "COMPARE_1D_RESOLUTION", "sxtdeg")

const S_REFERENCE = parse(Float64, get(ENV, "COMPARE_1D_S_REFERENCE", "35.0"))
const ρ₀ = parse(Float64, get(ENV, "COMPARE_1D_REFERENCE_DENSITY", "1035.0"))
const cₚ = parse(Float64, get(ENV, "COMPARE_1D_HEAT_CAPACITY", "1000.0"))

const NOLEAP_MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
const SECONDS_PER_DAY = 86400.0
const SECONDS_PER_YEAR = sum(NOLEAP_MONTH_DAYS) * SECONDS_PER_DAY
const CUMULATIVE_MONTH_SECONDS = cumsum(vcat(0, NOLEAP_MONTH_DAYS)) .* SECONDS_PER_DAY
const RYF_YEAR_DAYS = 365.0

run_number(path::AbstractString) = begin
    m = match(r"_run(\d+)\.jld2$", basename(path))
    isnothing(m) ? -1 : parse(Int, m.captures[1])
end

function with_trailing_slash(path::AbstractString)
    return endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator
end

function numeric_timeseries_keys(group)
    return sort(parse.(Int, filter(k -> tryparse(Int, k) !== nothing, collect(keys(group)))))
end

function trim_profile(raw, nz)
    values = collect(vec(raw))
    length(values) == nz && return Float64.(values)

    if length(values) < nz
        error("Profile length $(length(values)) is smaller than target Nz=$nz.")
    end

    halo = fld(length(values) - nz, 2)
    return Float64.(values[halo+1:halo+nz])
end

function model_integral_files(path::AbstractString, resolution::AbstractString)
    search_paths = String[]
    push!(search_paths, path)

    saved_path = joinpath(path, "saved")
    isdir(saved_path) && pushfirst!(search_paths, saved_path)

    for search_path in search_paths
        combined = glob("combined_global_*tot*$(resolution)*_RYF_run*.jld2", search_path)
        combined = filter(f -> !occursin("_rank", basename(f)), combined)

        if !isempty(combined)
            sort!(combined; by = run_number)
            return combined
        end
    end

    raw = glob("global_tot_integrals_$(resolution)_RYF_run*.jld2", path)
    raw = filter(f -> !occursin("_rank", basename(f)), raw)
    sort!(raw; by = run_number)
    return raw
end

function load_model_integral_series(path::AbstractString, resolution::AbstractString)
    files = model_integral_files(path, resolution)
    isempty(files) && error("No global integral files found for resolution=$(resolution) in $(path).")

    records_by_time = Dict{Float64, NamedTuple{(:run, :T, :S, :V, :Tz, :Sz, :Vz), Tuple{Int, Float64, Float64, Float64, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}()
    z_centers = Float64[]
    z_faces = Float64[]

    for file in files
        run = run_number(file)
        jldopen(file, "r") do data
            grid = with_logger(NullLogger()) do
                data["serialized/grid"]
            end
            underlying = hasproperty(grid, :underlying_grid) ? grid.underlying_grid : grid

            if isempty(z_centers)
                append!(z_centers, Float64.(collect(underlying.z.cᵃᵃᶜ)))
                append!(z_faces, Float64.(collect(underlying.z.cᵃᵃᶠ)))
            end

            iterations = numeric_timeseries_keys(data["timeseries/t"])
            nz = length(z_centers)

            for iter in iterations
                t = Float64(data["timeseries/t/$(iter)"])
                record = (
                    run = run,
                    T = Float64(only(data["timeseries/T_totintegral/$(iter)"])),
                    S = Float64(only(data["timeseries/S_totintegral/$(iter)"])),
                    V = Float64(only(data["timeseries/total_volume_c/$(iter)"])),
                    Tz = trim_profile(data["timeseries/T_vertintegral/$(iter)"][1, 1, :], nz),
                    Sz = trim_profile(data["timeseries/S_vertintegral/$(iter)"][1, 1, :], nz),
                    Vz = trim_profile(data["timeseries/vert_volume_c/$(iter)"][1, 1, :], nz)
                )

                existing = get(records_by_time, t, nothing)
                if isnothing(existing) || run >= existing.run
                    records_by_time[t] = record
                end
            end
        end
    end

    times = sort(collect(keys(records_by_time)))
    isempty(times) && error("No model timesteps found after merging integral files.")

    nt = length(times)
    nz = length(z_centers)

    T_total = Vector{Float64}(undef, nt)
    S_total = Vector{Float64}(undef, nt)
    V_total = Vector{Float64}(undef, nt)
    Tz = Matrix{Float64}(undef, nt, nz)
    Sz = Matrix{Float64}(undef, nt, nz)
    Vz = Matrix{Float64}(undef, nt, nz)

    for (n, t) in enumerate(times)
        record = records_by_time[t]
        T_total[n] = record.T
        S_total[n] = record.S
        V_total[n] = record.V
        Tz[n, :] = record.Tz
        Sz[n, :] = record.Sz
        Vz[n, :] = record.Vz
    end

    return (; times, z_centers, z_faces, T_total, S_total, V_total, Tz, Sz, Vz)
end

function month_of_year_and_start(t::Real)
    year_index = floor(Int, t / SECONDS_PER_YEAR)
    second_of_year = t - year_index * SECONDS_PER_YEAR
    month_of_year = searchsortedlast(CUMULATIVE_MONTH_SECONDS, second_of_year + eps(second_of_year))
    month_of_year = clamp(month_of_year, 1, 12)
    month_start = year_index * SECONDS_PER_YEAR + CUMULATIVE_MONTH_SECONDS[month_of_year]
    month_stop = year_index * SECONDS_PER_YEAR + CUMULATIVE_MONTH_SECONDS[month_of_year + 1]
    return year_index, month_of_year, month_start, month_stop
end

function split_interval_by_month(t0::Real, t1::Real)
    t1 <= t0 && return Tuple{Int, Int, Float64, Float64}[]

    pieces = Tuple{Int, Int, Float64, Float64}[]
    current = Float64(t0)
    stop = Float64(t1)

    while current < stop
        year_index, month_of_year, month_start, month_stop = month_of_year_and_start(current)
        piece_stop = min(stop, month_stop)
        push!(pieces, (year_index, month_of_year, current, piece_stop))
        current = piece_stop
    end

    return pieces
end

ryf_dates() = Date.(2001, 1:12, 1)

function ryf_month_starts_days()
    return Float64.(cumsum((0, NOLEAP_MONTH_DAYS[1:end-1]...)))
end

function ryf_month_ticks_days(xlimits_days::Tuple{Float64, Float64})
    month_starts = ryf_month_starts_days()
    month_labels = Dates.format.(ryf_dates(), "u")

    xmin, xmax = xlimits_days
    start_year = floor(Int, xmin / sum(NOLEAP_MONTH_DAYS))
    end_year = ceil(Int, xmax / sum(NOLEAP_MONTH_DAYS))

    positions = Float64[]
    labels = String[]

    for year in start_year:end_year
        offset = year * sum(NOLEAP_MONTH_DAYS)
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

allocate_like(::Number) = 0.0
allocate_like(x::AbstractVector) = zeros(Float64, length(x))

function monthly_average_scalar(times::AbstractVector{<:Real}, values::AbstractVector{<:Real})
    length(times) == length(values) || error("Times and scalar values must have the same length.")
    length(times) >= 2 || error("Need at least two model samples for dt-weighted monthly averages.")

    weighted_sum = Dict{Int, Float64}()
    total_dt = Dict{Int, Float64}()

    for n in 1:length(times)-1
        for (year_index, month_of_year, piece_start, piece_stop) in split_interval_by_month(times[n], times[n+1])
            bin = 12 * year_index + month_of_year
            dt = piece_stop - piece_start
            weighted_sum[bin] = get(weighted_sum, bin, 0.0) + values[n] * dt
            total_dt[bin] = get(total_dt, bin, 0.0) + dt
        end
    end

    bins = sort(collect(keys(weighted_sum)))
    means = [weighted_sum[bin] / total_dt[bin] for bin in bins]
    years = [fld(bin - 1, 12) for bin in bins]
    months = [mod1(bin, 12) for bin in bins]

    return (; bins, years, months, means, total_dt = [total_dt[bin] for bin in bins])
end

function monthly_average_profiles(times::AbstractVector{<:Real}, values::AbstractMatrix{<:Real})
    size(values, 1) == length(times) || error("Times and profile rows must have the same length.")
    length(times) >= 2 || error("Need at least two model samples for dt-weighted monthly averages.")

    weighted_sum = Dict{Int, Vector{Float64}}()
    total_dt = Dict{Int, Float64}()

    for n in 1:length(times)-1
        row = Float64.(vec(values[n, :]))
        for (year_index, month_of_year, piece_start, piece_stop) in split_interval_by_month(times[n], times[n+1])
            bin = 12 * year_index + month_of_year
            dt = piece_stop - piece_start
            if !haskey(weighted_sum, bin)
                weighted_sum[bin] = allocate_like(row)
            end
            weighted_sum[bin] .+= row .* dt
            total_dt[bin] = get(total_dt, bin, 0.0) + dt
        end
    end

    bins = sort(collect(keys(weighted_sum)))
    nz = size(values, 2)
    means = Matrix{Float64}(undef, length(bins), nz)

    for (n, bin) in enumerate(bins)
        means[n, :] = weighted_sum[bin] ./ total_dt[bin]
    end

    years = [fld(bin - 1, 12) for bin in bins]
    months = [mod1(bin, 12) for bin in bins]

    return (; bins, years, months, means, total_dt = [total_dt[bin] for bin in bins])
end

function load_woa_monthly_fields()
    arch = CPU()
    T_fields = Vector{Any}(undef, 12)
    S_fields = Vector{Any}(undef, 12)

    for month in 1:12
        date = DateTime(2018, month, 1)
        T_fields[month] = Field(Metadatum(:temperature; date, dataset=WOAMonthly()), arch; inpainting=nothing, cache_inpainted_data=false)
        S_fields[month] = Field(Metadatum(:salinity; date, dataset=WOAMonthly()), arch; inpainting=nothing, cache_inpainted_data=false)
    end

    return T_fields, S_fields
end

function wet_clean_fields(T, S)
    grid = T.grid
    T_raw = Float64.(Array(interior(T)))
    S_raw = Float64.(Array(interior(S)))

    wet = isfinite.(T_raw) .& isfinite.(S_raw)
    T_clean = ifelse.(wet, T_raw, 0.0)
    S_clean = ifelse.(wet, S_raw, 0.0)
    wet_clean = Float64.(wet)

    T_field = CenterField(grid)
    S_field = CenterField(grid)
    wet_field = CenterField(grid)

    set!(T_field, T_clean)
    set!(S_field, S_clean)
    set!(wet_field, wet_clean)

    return T_field, S_field, wet_field
end

function scalar_integral(field)
    integral = Field(Integral(field, dims=(1, 2, 3)))
    compute!(integral)
    return Float64(interior(integral)[1, 1, 1])
end

function vertical_integral(field)
    integral = Field(Integral(field, dims=(1, 2)))
    compute!(integral)
    return Float64.(vec(Array(interior(integral))))
end

function layer_thicknesses(z_faces::AbstractVector{<:Real})
    return diff(Float64.(z_faces))
end

function layer_overlaps(z_faces::AbstractVector{<:Real}, lower_bound::Real, upper_bound::Real)
    faces = Float64.(z_faces)
    lower = Float64(lower_bound)
    upper = Float64(upper_bound)

    lower < upper || error("The lower depth bound must be smaller than the upper depth bound.")
    all(diff(faces) .> 0) || error("Vertical faces must be strictly increasing.")

    return max.(0.0, min.(faces[2:end], upper) .- max.(faces[1:end-1], lower))
end

function integrate_horizontal_profiles(profiles::AbstractMatrix, z_faces, lower_bound, upper_bound)
    overlaps = layer_overlaps(z_faces, lower_bound, upper_bound)
    size(profiles, 2) == length(overlaps) || error("Profile columns must match the number of vertical layers.")
    return Float64.(profiles) * overlaps
end

function integrate_layer_totals(profiles::AbstractMatrix, z_faces, lower_bound, upper_bound)
    thicknesses = layer_thicknesses(z_faces)
    overlaps = layer_overlaps(z_faces, lower_bound, upper_bound)
    size(profiles, 2) == length(overlaps) || error("Profile columns must match the number of vertical layers.")
    return Float64.(profiles) * (overlaps ./ thicknesses)
end

function overlap_conservative_regrid_density(source_density, source_faces, target_faces)
    source_density = Float64.(source_density)
    source_faces = Float64.(source_faces)
    target_faces = Float64.(target_faces)

    ns = length(source_density)
    nt = length(target_faces) - 1
    length(source_faces) == ns + 1 || error("source_faces length must equal source_density length + 1.")

    result = zeros(Float64, nt)

    for j in 1:nt
        a = target_faces[j]
        b = target_faces[j + 1]
        Δ = b - a
        Δ <= 0 && error("Target faces must be strictly increasing.")

        total = 0.0
        for i in 1:ns
            left = max(a, source_faces[i])
            right = min(b, source_faces[i + 1])
            overlap = right - left
            overlap > 0 && (total += source_density[i] * overlap)
        end

        result[j] = total / Δ
    end

    return result
end

function conservative_vertical_regrid_density(source_density, source_faces, target_faces)
    return overlap_conservative_regrid_density(source_density, source_faces, target_faces)
end

function load_woa_monthly_reference()
    T_fields, S_fields = load_woa_monthly_fields()

    reference = Dict{Symbol, Any}()
    global_T = zeros(Float64, 12)
    global_S = zeros(Float64, 12)
    global_V = zeros(Float64, 12)
    Tz = nothing
    Sz = nothing
    Vz = nothing
    z_centers = Float64[]
    z_faces = Float64[]

    for month in 1:12
        T_clean, S_clean, wet = wet_clean_fields(T_fields[month], S_fields[month])

        if month == 1
            nz = size(interior(T_clean), 3)
            Hz = T_clean.grid.Hz
            append!(z_centers, Float64.(collect(T_clean.grid.z.cᵃᵃᶜ[Hz+1:Hz+nz])))
            append!(z_faces, Float64.(collect(T_clean.grid.z.cᵃᵃᶠ[Hz+1:Hz+nz+1])))
            Tz = Matrix{Float64}(undef, 12, nz)
            Sz = Matrix{Float64}(undef, 12, nz)
            Vz = Matrix{Float64}(undef, 12, nz)
        end

        global_T[month] = scalar_integral(T_clean)
        global_S[month] = scalar_integral(S_clean)
        global_V[month] = scalar_integral(wet)
        Tz[month, :] = vertical_integral(T_clean)
        Sz[month, :] = vertical_integral(S_clean)
        Vz[month, :] = vertical_integral(wet)
    end

    reference[:z_centers] = z_centers
    reference[:z_faces] = z_faces
    reference[:T_total] = global_T
    reference[:S_total] = global_S
    reference[:V_total] = global_V
    reference[:Tz] = Tz
    reference[:Sz] = Sz
    reference[:Vz] = Vz

    return reference
end

function remap_vertical_integral_profiles_to_model_grid(source_profiles::AbstractMatrix, source_faces, target_faces)
    source_faces = Float64.(source_faces)
    target_faces = Float64.(target_faces)
    Δz_source = layer_thicknesses(source_faces)
    Δz_target = layer_thicknesses(target_faces)

    nz_target = length(Δz_target)
    target_profiles = Matrix{Float64}(undef, size(source_profiles, 1), nz_target)

    for month in axes(source_profiles, 1)
        q = Float64.(source_profiles[month, :]) ./ Δz_source
        target_profiles[month, :] = conservative_vertical_regrid_density(q, source_faces, target_faces) .* Δz_target
    end

    return target_profiles
end

function remap_woa_profiles_to_model_grid(woa, model_z_faces)
    source_faces = woa[:z_faces]
    target_faces = Float64.(model_z_faces)

    OHCz_native = global_ohc.(woa[:Tz])
    OFWCz_native = global_ofwc.(woa[:Sz], woa[:Vz])
    Vz_target = remap_vertical_integral_profiles_to_model_grid(woa[:Vz], source_faces, target_faces)
    OHCz_target = remap_vertical_integral_profiles_to_model_grid(OHCz_native, source_faces, target_faces)
    OFWCz_target = remap_vertical_integral_profiles_to_model_grid(OFWCz_native, source_faces, target_faces)

    return (; OHC = OHCz_target, OFWC = OFWCz_target, Vz = Vz_target)
end

global_ohc(T_integral) = ρ₀ * cₚ * T_integral
global_ofwc(S_integral, volume) = ρ₀ * (S_REFERENCE * volume - S_integral) / S_REFERENCE

function tiled_monthly_reference(reference_12::AbstractVector, months::AbstractVector{<:Integer})
    return [reference_12[month] for month in months]
end

function tiled_monthly_reference(reference_12::AbstractMatrix, months::AbstractVector{<:Integer})
    out = Matrix{Float64}(undef, length(months), size(reference_12, 2))
    for (n, month) in enumerate(months)
        out[n, :] = reference_12[month, :]
    end
    return out
end

function plot_series_with_markers!(ax, x, y; kwargs...)
    lines!(ax, x, y; kwargs...)
    scatter!(ax, x, y; color = get(kwargs, :color, :black), markersize = 10)
end

function padded_xlim(x::AbstractVector{<:Real})
    xmin = minimum(x)
    xmax = maximum(x)
    if isapprox(xmin, xmax; atol=0, rtol=0)
        return (xmin - 15, xmax + 15)
    end
    padding = 0.05 * (xmax - xmin)
    return (xmin - padding, xmax + padding)
end

function save_comparison_figure(path, time_years, model_global, woa_global, delta_global)
    fig = Figure(size=(1200, 900))

    ax1 = Axis(fig[1, 1], title="Monthly Mean OHC", xlabel="Year", ylabel="J")
    ax2 = Axis(fig[1, 2], title="Monthly Mean OFWC", xlabel="Year", ylabel="kg")
    ax3 = Axis(fig[2, 1], title="Model - WOA OHC", xlabel="Year", ylabel="J")
    ax4 = Axis(fig[2, 2], title="Model - WOA OFWC", xlabel="Year", ylabel="kg")

    plot_series_with_markers!(ax1, time_years, model_global.OHC; color=:dodgerblue4, label="Model")
    plot_series_with_markers!(ax1, time_years, woa_global.OHC; color=:black, label="WOA")
    axislegend(ax1, position=:rb)

    plot_series_with_markers!(ax2, time_years, model_global.OFWC; color=:dodgerblue4, label="Model")
    plot_series_with_markers!(ax2, time_years, woa_global.OFWC; color=:black, label="WOA")
    axislegend(ax2, position=:rb)

    plot_series_with_markers!(ax3, time_years, delta_global.OHC; color=:firebrick)
    plot_series_with_markers!(ax4, time_years, delta_global.OFWC; color=:steelblue)

    xlimits = padded_xlim(time_years)

    for ax in (ax1, ax2, ax3, ax4)
        xlims!(ax, xlimits)
    end

    save(path, fig, px_per_unit=2)
    return nothing
end

function finite_maxabs(A)
    values = filter(isfinite, vec(Float64.(A)))
    isempty(values) && return 1.0
    return maximum(abs, values)
end

function save_depth_comparison_figure(path, time_years, depth, model_vertical, woa_vertical, delta_vertical)
    fig = Figure(size=(1600, 1000))

    xlimits = padded_xlim(time_years)

    ohc_cr = finite_maxabs(vcat(model_vertical.OHC, woa_vertical.OHC, delta_vertical.OHC))
    ofwc_cr = finite_maxabs(vcat(model_vertical.OFWC, woa_vertical.OFWC, delta_vertical.OFWC))

    ax11 = Axis(fig[1, 1], title="Model OHC", xlabel="Year", ylabel="Depth (m)")
    ax12 = Axis(fig[1, 2], title="WOA OHC", xlabel="Year", ylabel="Depth (m)")
    ax13 = Axis(fig[1, 3], title="Model - WOA OHC", xlabel="Year", ylabel="Depth (m)")
    ax21 = Axis(fig[2, 1], title="Model OFWC", xlabel="Year", ylabel="Depth (m)")
    ax22 = Axis(fig[2, 2], title="WOA OFWC", xlabel="Year", ylabel="Depth (m)")
    ax23 = Axis(fig[2, 3], title="Model - WOA OFWC", xlabel="Year", ylabel="Depth (m)")

    hm11 = heatmap!(ax11, time_years, depth, model_vertical.OHC, colorrange=(-ohc_cr, ohc_cr), colormap=:balance)
    hm12 = heatmap!(ax12, time_years, depth, woa_vertical.OHC, colorrange=(-ohc_cr, ohc_cr), colormap=:balance)
    hm13 = heatmap!(ax13, time_years, depth, delta_vertical.OHC, colorrange=(-ohc_cr, ohc_cr), colormap=:balance)
    hm21 = heatmap!(ax21, time_years, depth, model_vertical.OFWC, colorrange=(-ofwc_cr, ofwc_cr), colormap=:balance)
    hm22 = heatmap!(ax22, time_years, depth, woa_vertical.OFWC, colorrange=(-ofwc_cr, ofwc_cr), colormap=:balance)
    hm23 = heatmap!(ax23, time_years, depth, delta_vertical.OFWC, colorrange=(-ofwc_cr, ofwc_cr), colormap=:balance)

    for ax in (ax11, ax12, ax13, ax21, ax22, ax23)
        xlims!(ax, xlimits)
        ylims!(ax, -1000, 0)
    end

    Colorbar(fig[1, 4], hm13, label="OHC anomaly (J)")
    Colorbar(fig[2, 4], hm23, label="OFWC anomaly (kg)")

    save(path, fig, px_per_unit=2)
    return nothing
end

function main()
    mkpath(ANALYSIS_OUTPUT_PATH)
    mkpath(FIGDIR)

    @info "Loading model global and vertical integrals" output_path = OUTPUT_PATH resolution = RESOLUTION
    model = load_model_integral_series(OUTPUT_PATH, RESOLUTION)

    @info "Computing dt-weighted monthly model means"
    monthly_T_total = monthly_average_scalar(model.times, model.T_total)
    monthly_S_total = monthly_average_scalar(model.times, model.S_total)
    monthly_V_total = monthly_average_scalar(model.times, model.V_total)
    monthly_Tz = monthly_average_profiles(model.times, model.Tz)
    monthly_Sz = monthly_average_profiles(model.times, model.Sz)
    monthly_Vz = monthly_average_profiles(model.times, model.Vz)

    bins = monthly_T_total.bins
    monthly_S_total.bins == bins || error("Monthly scalar bins do not align for T and S.")
    monthly_V_total.bins == bins || error("Monthly scalar bins do not align for T and V.")
    monthly_Tz.bins == bins || error("Monthly profile bins do not align for Tz.")
    monthly_Sz.bins == bins || error("Monthly profile bins do not align for Sz.")
    monthly_Vz.bins == bins || error("Monthly profile bins do not align for Vz.")

    month_of_year = monthly_T_total.months
    year_index = monthly_T_total.years
    time_years = (Float64.(year_index) .+ (Float64.(month_of_year) .- 0.5) ./ 12)
    time_days = year_index .* RYF_YEAR_DAYS .+ ryf_month_starts_days()[month_of_year] .+ 0.5 .* NOLEAP_MONTH_DAYS[month_of_year]

    model_vertical = (
        T_integral = monthly_Tz.means,
        S_integral = monthly_Sz.means,
        volume = monthly_Vz.means,
        OHC = global_ohc.(monthly_Tz.means),
        OFWC = global_ofwc.(monthly_Sz.means, monthly_Vz.means)
    )

    @info "Loading WOA monthly climatology on the native WOA grid"
    woa_native = load_woa_monthly_reference()

    woa_lower_bound, woa_upper_bound = extrema(woa_native[:z_faces])

    @info "Matching model global integrals to the WOA depth range" woa_lower_bound woa_upper_bound
    model_T_total = integrate_horizontal_profiles(monthly_Tz.means, model.z_faces, woa_lower_bound, woa_upper_bound)
    model_S_total = integrate_horizontal_profiles(monthly_Sz.means, model.z_faces, woa_lower_bound, woa_upper_bound)
    model_V_total = integrate_layer_totals(monthly_Vz.means, model.z_faces, woa_lower_bound, woa_upper_bound)

    model_global = (
        T_integral = model_T_total,
        S_integral = model_S_total,
        volume = model_V_total,
        OHC = global_ohc.(model_T_total),
        OFWC = global_ofwc.(model_S_total, model_V_total)
    )

    @info "Conservatively remapping native-grid WOA vertical profiles to the model z-faces"
    woa_model_grid_vertical = remap_woa_profiles_to_model_grid(woa_native, model.z_faces)

    woa_global = (
        T_integral = tiled_monthly_reference(woa_native[:T_total], month_of_year),
        S_integral = tiled_monthly_reference(woa_native[:S_total], month_of_year),
        volume = tiled_monthly_reference(woa_native[:V_total], month_of_year),
        OHC = tiled_monthly_reference(global_ohc.(woa_native[:T_total]), month_of_year),
        OFWC = tiled_monthly_reference(global_ofwc.(woa_native[:S_total], woa_native[:V_total]), month_of_year)
    )

    model_global = merge(model_global, (
        OHC = model_global.OHC .- first(model_global.OHC),
        OFWC = model_global.OFWC .- first(model_global.OFWC)
    ))

    woa_global = merge(woa_global, (
        OHC = woa_global.OHC .- first(woa_global.OHC),
        OFWC = woa_global.OFWC .- first(woa_global.OFWC)
    ))

    woa_vertical = (
        volume = tiled_monthly_reference(woa_model_grid_vertical.Vz, month_of_year),
        OHC = tiled_monthly_reference(woa_model_grid_vertical.OHC, month_of_year),
        OFWC = tiled_monthly_reference(woa_model_grid_vertical.OFWC, month_of_year)
    )

    model_vertical = merge(model_vertical, (
        OHC = model_vertical.OHC .- model_vertical.OHC[1:1, :],
        OFWC = model_vertical.OFWC .- model_vertical.OFWC[1:1, :]
    ))

    woa_vertical = merge(woa_vertical, (
        OHC = woa_vertical.OHC .- woa_vertical.OHC[1:1, :],
        OFWC = woa_vertical.OFWC .- woa_vertical.OFWC[1:1, :]
    ))

    delta_global = (
        OHC = model_global.OHC .- woa_global.OHC,
        OFWC = model_global.OFWC .- woa_global.OFWC
    )

    delta_vertical = (
        OHC = model_vertical.OHC .- woa_vertical.OHC,
        OFWC = model_vertical.OFWC .- woa_vertical.OFWC
    )

    output_file = joinpath(ANALYSIS_OUTPUT_PATH, "compare_1d_WOA_$(RESOLUTION).jld2")
    fig_file = joinpath(FIGDIR, "compare_1d_WOA_global_$(RESOLUTION).png")
    fig_z_file = joinpath(FIGDIR, "compare_1d_WOA_depth_$(RESOLUTION).png")

    @info "Saving WOA comparison dataset" output_file
    jldsave(output_file;
            resolution = RESOLUTION,
            reference_salinity = S_REFERENCE,
            reference_density = ρ₀,
            heat_capacity = cₚ,
            month_bin = bins,
            month_of_year,
            year_index,
            time_days,
            time_years,
            model_z_centers = model.z_centers,
            model_z_faces = model.z_faces,
            woa_native_z_centers = woa_native[:z_centers],
            woa_native_z_faces = woa_native[:z_faces],
            woa_lower_bound,
            woa_upper_bound,
            model_global,
            model_vertical,
            woa_global,
            woa_vertical,
            delta_global,
            delta_vertical)

    @info "Saving quick-look global comparison figure" fig_file
    save_comparison_figure(fig_file, time_years, model_global, woa_global, delta_global)

    @info "Saving depth-resolved WOA comparison figure" fig_z_file
    save_depth_comparison_figure(fig_z_file, time_years, model.z_centers, model_vertical, woa_vertical, delta_vertical)

    @info "Finished building 1D WOA comparison machinery" output_file fig_file fig_z_file nmonths = length(time_years)
end

main()
