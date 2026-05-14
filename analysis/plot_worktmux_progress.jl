using Printf

const HAS_CAIROMAKIE = try
    @eval using CairoMakie
    true
catch err
    @warn "CairoMakie is unavailable; the script will write an SVG instead." exception=(err, catch_backtrace())
    false
end

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_LOG = joinpath(ROOT_DIR, "experiments", "submission_scripts", "logs",
                             "worktmux_spartan-gpgpu172.log")
const DEFAULT_OUTPUT = joinpath(ROOT_DIR, "figures", "worktmux_spartan-gpgpu172_progress.png")
const DEFAULT_SEA_ICE_OUTPUT = joinpath(ROOT_DIR, "figures", "worktmux_spartan-gpgpu172_sea_ice.png")
const DEFAULT_NAN_CHECK_OUTPUT = joinpath(ROOT_DIR, "figures", "worktmux_spartan-gpgpu172_nan_check_extrema.png")

const NUMBER = raw"(?:NaN|[-+]?Inf|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"

function strip_terminal_codes(line)
    line = replace(line, r"\e\][^\a]*(?:\a|\e\\)" => "")
    line = replace(line, r"\e\[[0-?]*[ -/]*[@-~]" => "")
    return replace(line, '\r' => "")
end

function parse_number(text)
    str = strip(text)
    return str == "NaN" ? NaN : parse(Float64, str)
end

function parse_duration_seconds(value, unit)
    t = parse_number(value)
    unit = lowercase(unit)

    if startswith(unit, "second")
        return t
    elseif startswith(unit, "minute")
        return 60 * t
    elseif startswith(unit, "hour")
        return 60 * 60 * t
    elseif startswith(unit, "day")
        return 24 * 60 * 60 * t
    else
        error("Unknown time unit: $unit")
    end
end

parse_time_days(value, unit) = parse_duration_seconds(value, unit) / (24 * 60 * 60)

function push_missing_record!(series)
    for ys in values(series)
        push!(ys, NaN)
    end
end

function set_current_value!(series, name, value, records)
    ys = get!(series, name) do
        fill(NaN, records)
    end

    if length(ys) < records
        append!(ys, fill(NaN, records - length(ys)))
    end

    ys[end] = value
    return value
end

normalize_extrema_name(name) = strip(name) == "." ? "eta" : strip(name)
extrema_series_name(name, kind) = "extrema(" * name * ")_" * kind
nan_check_series_name(name, kind) = "nan_check(" * name * ")_" * kind

function parse_progress_log(path)
    series = Dict{String, Vector{Float64}}()

    record_re = Regex("time:\\s*($NUMBER)\\s*(seconds?|minutes?|hours?|days?),\\s*iteration:\\s*(\\d+)")
    max_abs_u_re = Regex("max\\|u\\|:\\s*\\(($NUMBER),\\s*($NUMBER),\\s*($NUMBER)\\)")
    extrema_re = Regex("extrema\\(([^)]+)\\):\\s*\\(($NUMBER),\\s*($NUMBER)\\)")
    wall_time_re = Regex("wall time:\\s*($NUMBER)\\s*(seconds?|minutes?|hours?|days?)")
    wall_clock_re = Regex("Wall clock time:\\s*($NUMBER) days")
    sypd_re = Regex("SYPD:\\s*($NUMBER)")
    cfl_re = Regex("advective_cfl:\\s*($NUMBER)")

    records = 0

    for raw_line in eachline(path)
        line = strip_terminal_codes(raw_line)

        m = match(record_re, line)
        if m !== nothing
            records += 1
            push_missing_record!(series)

            set_current_value!(series, "time_days", parse_time_days(m.captures[1], m.captures[2]), records)
            set_current_value!(series, "iteration", parse(Float64, m.captures[3]), records)

            if (mu = match(max_abs_u_re, line)) !== nothing
                set_current_value!(series, "max_abs_u", parse_number(mu.captures[1]), records)
                set_current_value!(series, "max_abs_v", parse_number(mu.captures[2]), records)
                set_current_value!(series, "max_abs_w", parse_number(mu.captures[3]), records)
            end

            for em in eachmatch(extrema_re, line)
                name = normalize_extrema_name(em.captures[1])
                set_current_value!(series, extrema_series_name(name, "max"), parse_number(em.captures[2]), records)
                set_current_value!(series, extrema_series_name(name, "min"), parse_number(em.captures[3]), records)
            end

            if (wm = match(wall_time_re, line)) !== nothing
                set_current_value!(series, "wall_time_seconds",
                                   parse_duration_seconds(wm.captures[1], wm.captures[2]), records)
            end

            continue
        end

        records == 0 && continue

        if (m = match(wall_clock_re, line)) !== nothing
            set_current_value!(series, "wall_clock_days", parse_number(m.captures[1]), records)
        elseif (m = match(sypd_re, line)) !== nothing
            set_current_value!(series, "SYPD", parse_number(m.captures[1]), records)
        elseif (m = match(cfl_re, line)) !== nothing
            set_current_value!(series, "advective_cfl", parse_number(m.captures[1]), records)
        elseif occursin("sea ice at advective_cfl max:", line)
            values = [parse_number(m.match) for m in eachmatch(Regex(NUMBER), line)]
            length(values) < 14 && continue

            set_current_value!(series, "sea_ice_h", values[1], records)
            set_current_value!(series, "sea_ice_area", values[2], records)
            set_current_value!(series, "sea_ice_qio", values[3], records)
            set_current_value!(series, "sea_ice_Ssurf", values[4], records)
            set_current_value!(series, "sea_ice_Sice", values[5], records)
            set_current_value!(series, "sea_ice_Jsio", values[6], records)
            set_current_value!(series, "sea_ice_Qio", values[7], records)
            set_current_value!(series, "sea_ice_Qfrazil", values[8], records)
            set_current_value!(series, "sea_ice_tauio_x", values[9], records)
            set_current_value!(series, "sea_ice_tauio_y", values[10], records)
            set_current_value!(series, "net_ocean_JT", values[11], records)
            set_current_value!(series, "net_ocean_JS", values[12], records)
            set_current_value!(series, "net_ocean_tau_x", values[13], records)
            set_current_value!(series, "net_ocean_tau_y", values[14], records)
        end
    end

    return series
end

function parse_nan_check_extrema_log(path)
    series = Dict{String, Vector{Float64}}()

    header_re = Regex("NaN-check extrema at iteration\\s*(\\d+),\\s*time\\s*($NUMBER)\\s*(seconds?|minutes?|hours?|days?)")
    diagnostic_re = Regex("^\\S\\s*(.+?):\\s*first non-finite\\s*=.*?;\\s*finite max\\s*=\\s*($NUMBER).*?;\\s*finite min\\s*=\\s*($NUMBER)")

    records = 0

    for raw_line in eachline(path)
        line = strip_terminal_codes(raw_line)

        if (m = match(header_re, line)) !== nothing
            records += 1
            push_missing_record!(series)

            set_current_value!(series, "time_days", parse_time_days(m.captures[2], m.captures[3]), records)
            set_current_value!(series, "iteration", parse(Float64, m.captures[1]), records)
            continue
        end

        records == 0 && continue

        m = match(diagnostic_re, line)
        m === nothing && continue

        name = strip(m.captures[1])
        set_current_value!(series, nan_check_series_name(name, "max"), parse_number(m.captures[2]), records)
        set_current_value!(series, nan_check_series_name(name, "min"), parse_number(m.captures[3]), records)
    end

    return series
end

function parse_first_nonfinite_event(path)
    header_re = Regex("NaN-check extrema at iteration\\s*(\\d+),\\s*time\\s*($NUMBER)\\s*(seconds?|minutes?|hours?|days?)")
    nonfinite_re = Regex("^\\S\\s*(.+?):\\s*first non-finite\\s*=\\s*([^;]+);")

    current_iteration = NaN
    current_time_days = NaN

    for raw_line in eachline(path)
        line = strip_terminal_codes(raw_line)

        if (m = match(header_re, line)) !== nothing
            current_iteration = parse(Float64, m.captures[1])
            current_time_days = parse_time_days(m.captures[2], m.captures[3])
            continue
        end

        m = match(nonfinite_re, line)
        m === nothing && continue

        first_nonfinite = strip(m.captures[2])
        lowercase(first_nonfinite) == "none" && continue

        return (
            iteration = current_iteration,
            time_days = current_time_days,
            variable = strip(m.captures[1]),
            value = first_nonfinite,
        )
    end

    return nothing
end

function extrema_variables(series)
    found = Set{String}()

    for name in keys(series)
        m = match(r"^extrema\((.*)\)_(?:max|min)$", name)
        m === nothing && continue
        push!(found, m.captures[1])
    end

    preferred_order = ["T", "S", "\u03b7", "eta"]
    ordered = String[]
    for name in preferred_order
        name in found && push!(ordered, name)
    end

    for name in sort!(collect(found))
        name in preferred_order && continue
        push!(ordered, name)
    end

    return ordered
end

function nan_check_rank(name)
    !occursin(".", name) && return 1

    prefixes = [
        "net_ocean_fluxes.",
        "atmosphere_ocean_fluxes.",
        "sea_ice_ocean_fluxes.",
        "ocean_radiation_fluxes.",
        "sea_ice.",
    ]

    idx = findfirst(prefix -> startswith(name, prefix), prefixes)
    return idx === nothing ? length(prefixes) + 2 : idx + 1
end

function nan_check_variables(series)
    found = Set{String}()

    for name in keys(series)
        m = match(r"^nan_check\((.*)\)_(?:max|min)$", name)
        m === nothing && continue
        push!(found, m.captures[1])
    end

    return sort!(collect(found), by = name -> (nan_check_rank(name), name))
end

function series_has_values(series, name)
    return haskey(series, name) && any(isfinite, series[name])
end

function extrema_panes(series)
    panes = Tuple{String, String}[]

    for name in extrema_variables(series)
        max_name = extrema_series_name(name, "max")
        min_name = extrema_series_name(name, "min")

        if series_has_values(series, max_name)
            push!(panes, (max_name, "max extrema($name)"))
        end

        if series_has_values(series, min_name)
            push!(panes, (min_name, "min extrema($name)"))
        end
    end

    return panes
end

function nan_check_panes(series)
    panes = Tuple{String, String}[]

    for name in nan_check_variables(series)
        max_name = nan_check_series_name(name, "max")
        min_name = nan_check_series_name(name, "min")

        if series_has_values(series, max_name)
            push!(panes, (max_name, "NaN-check max $name"))
        end

        if series_has_values(series, min_name)
            push!(panes, (min_name, "NaN-check min $name"))
        end
    end

    return panes
end

function progress_panes(series)
    panes = [
        ("iteration", "iteration"),
        ("wall_clock_days", "wall clock days"),
        ("wall_time_seconds", "wall time per report (s)"),
        ("SYPD", "SYPD"),
        ("advective_cfl", "advective CFL"),
        ("max_abs_u", "max abs u (m s^-1)"),
        ("max_abs_v", "max abs v (m s^-1)"),
        ("max_abs_w", "max abs w (m s^-1)"),
    ]

    return vcat(filter(p -> series_has_values(series, p[1]), panes), extrema_panes(series))
end

function sea_ice_panes(series)
    panes = [
        ("sea_ice_h", "sea ice h at CFL max (m)"),
        ("sea_ice_area", "sea ice area at CFL max"),
        ("sea_ice_qio", "sea ice qio at CFL max (m s^-1)"),
        ("sea_ice_Ssurf", "sea ice Ssurf at CFL max (g kg^-1)"),
        ("sea_ice_Sice", "sea ice Sice at CFL max (g kg^-1)"),
        ("sea_ice_Jsio", "sea ice Jsio at CFL max"),
        ("sea_ice_Qio", "sea ice Qio at CFL max (W m^-2)"),
        ("sea_ice_Qfrazil", "sea ice Qfrazil at CFL max (W m^-2)"),
        ("sea_ice_tauio_x", "sea ice tauio x at CFL max (N m^-2)"),
        ("sea_ice_tauio_y", "sea ice tauio y at CFL max (N m^-2)"),
        ("net_ocean_JT", "net ocean JT at CFL max"),
        ("net_ocean_JS", "net ocean JS at CFL max"),
        ("net_ocean_tau_x", "net ocean tau x at CFL max (m^2 s^-2)"),
        ("net_ocean_tau_y", "net ocean tau y at CFL max (m^2 s^-2)"),
    ]

    return filter(p -> series_has_values(series, p[1]), panes)
end

function finite_limits(x, y)
    mask = isfinite.(x) .& isfinite.(y)
    return x[mask], y[mask]
end

has_event_marker(event_time_days) = event_time_days !== nothing && isfinite(event_time_days)

function plot_progress_makie(series; output_path=DEFAULT_OUTPUT, panes=progress_panes(series),
                             title="Progress parsed from $(basename(DEFAULT_LOG))",
                             x_name="time_days", x_label="model time (days)",
                             event_time_days=nothing)
    x = series[x_name]
    isempty(panes) && error("No progress values found in log.")

    ncols = 2
    nrows = cld(length(panes), ncols)
    fig = Figure(size=(1500, 300nrows), fontsize=14)

    for (i, (name, label)) in enumerate(panes)
        row = cld(i, ncols)
        col = mod1(i, ncols)
        ax = Axis(fig[row, col], xlabel=x_label, ylabel=label, title=label)
        xf, yf = finite_limits(x, series[name])
        lines!(ax, xf, yf, color=:navy, linewidth=2)
        scatter!(ax, xf, yf, color=:navy, markersize=5)

        if has_event_marker(event_time_days)
            vlines!(ax, [event_time_days], color=:crimson, linestyle=:dash, linewidth=2)
        end
    end

    Label(fig[0, :], title, fontsize=22, tellwidth=false)
    save(output_path, fig)
    return output_path
end

svg_escape(text) = replace(replace(replace(text, "&" => "&amp;"), "<" => "&lt;"), ">" => "&gt;")

function nice_limits(values)
    vals = values[isfinite.(values)]
    isempty(vals) && return (0.0, 1.0)
    lo, hi = extrema(vals)
    if lo == hi
        delta = iszero(lo) ? 1.0 : 0.05 * abs(lo)
        return lo - delta, hi + delta
    end
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad
end

tick_values(lo, hi; n=4) = collect(range(lo, hi; length=n))

function fmt_tick(x)
    if !isfinite(x)
        return "NaN"
    elseif abs(x) >= 1e4 || (abs(x) < 1e-2 && x != 0)
        return @sprintf("%.2e", x)
    else
        return @sprintf("%.3g", x)
    end
end

function polyline_points(x, y, xmin, xmax, ymin, ymax, left, top, width, height)
    points = String[]
    for (xi, yi) in zip(x, y)
        if isfinite(xi) && isfinite(yi)
            px = left + width * (xi - xmin) / (xmax - xmin)
            py = top + height * (1 - (yi - ymin) / (ymax - ymin))
            push!(points, @sprintf("%.2f,%.2f", px, py))
        end
    end
    return join(points, " ")
end

function plot_progress_svg(series; output_path=replace(DEFAULT_OUTPUT, r"\.[^.]*$" => ".svg"),
                           panes=progress_panes(series),
                           title="Progress parsed from $(basename(DEFAULT_LOG))",
                           x_name="time_days", x_label="model time (days)",
                           event_time_days=nothing)
    x = series[x_name]
    isempty(panes) && error("No progress values found in log.")

    ncols = 2
    nrows = cld(length(panes), ncols)
    pane_w, pane_h = 700, 260
    margin_x, margin_y = 70, 70
    plot_w, plot_h = pane_w - 120, pane_h - 110
    width, height = ncols * pane_w, nrows * pane_h + 70
    xmin, xmax = nice_limits(x)

    open(output_path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$(width / 2)" y="35" text-anchor="middle" font-family="sans-serif" font-size="24" font-weight="700">$(svg_escape(title))</text>""")

        for (i, (name, label)) in enumerate(panes)
            row = cld(i, ncols) - 1
            col = mod1(i, ncols) - 1
            origin_x = col * pane_w
            origin_y = row * pane_h + 55
            left = origin_x + margin_x
            top = origin_y + margin_y - 20
            y = series[name]
            ymin, ymax = nice_limits(y)
            points = polyline_points(x, y, xmin, xmax, ymin, ymax, left, top, plot_w, plot_h)

            println(io, """<text x="$(left + plot_w / 2)" y="$(origin_y + 25)" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="700">$(svg_escape(label))</text>""")
            println(io, """<rect x="$left" y="$top" width="$plot_w" height="$plot_h" fill="#f8fafc" stroke="#cbd5e1"/>""")

            for tv in tick_values(ymin, ymax)
                py = top + plot_h * (1 - (tv - ymin) / (ymax - ymin))
                println(io, """<line x1="$left" y1="$py" x2="$(left + plot_w)" y2="$py" stroke="#e2e8f0"/>""")
                println(io, """<text x="$(left - 8)" y="$(py + 4)" text-anchor="end" font-family="sans-serif" font-size="11" fill="#475569">$(fmt_tick(tv))</text>""")
            end

            for tv in tick_values(xmin, xmax)
                px = left + plot_w * (tv - xmin) / (xmax - xmin)
                println(io, """<line x1="$px" y1="$top" x2="$px" y2="$(top + plot_h)" stroke="#e2e8f0"/>""")
                println(io, """<text x="$px" y="$(top + plot_h + 18)" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#475569">$(fmt_tick(tv))</text>""")
            end

            if has_event_marker(event_time_days)
                px = left + plot_w * (event_time_days - xmin) / (xmax - xmin)
                println(io, """<line x1="$px" y1="$top" x2="$px" y2="$(top + plot_h)" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 5"/>""")
            end

            println(io, """<polyline points="$points" fill="none" stroke="#0f172a" stroke-width="2"/>""")
            println(io, """<text x="$(left + plot_w / 2)" y="$(top + plot_h + 42)" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#334155">$(svg_escape(x_label))</text>""")
        end

        println(io, "</svg>")
    end

    return output_path
end

function plot_progress(series; output_path=DEFAULT_OUTPUT, panes=progress_panes(series),
                       title="Progress parsed from $(basename(DEFAULT_LOG))",
                       x_name="time_days", x_label="model time (days)",
                       event_time_days=nothing)
    if HAS_CAIROMAKIE
        return plot_progress_makie(series; output_path, panes, title, x_name, x_label, event_time_days)
    else
        return plot_progress_svg(series; output_path=replace(output_path, r"\.[^.]*$" => ".svg"),
                                 panes, title, x_name, x_label, event_time_days)
    end
end

function main()
    log_path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_LOG
    output_path = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_OUTPUT
    sea_ice_output_path = length(ARGS) >= 3 ? ARGS[3] : DEFAULT_SEA_ICE_OUTPUT
    nan_check_output_path = length(ARGS) >= 4 ? ARGS[4] : DEFAULT_NAN_CHECK_OUTPUT

    series = parse_progress_log(log_path)
    nan_check_series = parse_nan_check_extrema_log(log_path)
    first_nonfinite_event = parse_first_nonfinite_event(log_path)
    event_time_days = first_nonfinite_event === nothing ? nothing : first_nonfinite_event.time_days

    output = plot_progress(series; output_path, panes=progress_panes(series),
                           title="Progress parsed from $(basename(log_path))",
                           event_time_days)
    sea_ice_output = plot_progress(series; output_path=sea_ice_output_path, panes=sea_ice_panes(series),
                                   title="Sea ice diagnostics at advective CFL max",
                                   event_time_days)
    nan_check_output = nothing
    nan_check_pane_list = nan_check_panes(nan_check_series)
    if !isempty(get(nan_check_series, "time_days", Float64[])) && !isempty(nan_check_pane_list)
        nan_check_output = plot_progress(nan_check_series; output_path=nan_check_output_path,
                                         panes=nan_check_pane_list,
                                         title="NaN-check extrema parsed from $(basename(log_path))",
                                         event_time_days)
    end

    println("Parsed $(length(series["time_days"])) progress records.")
    println("Parsed $(length(get(nan_check_series, "time_days", Float64[]))) NaN-check extrema records.")
    if first_nonfinite_event !== nothing
        println("First non-finite detected at iteration $(first_nonfinite_event.iteration), time $(first_nonfinite_event.time_days) days: $(first_nonfinite_event.variable) = $(first_nonfinite_event.value)")
    else
        println("No non-finite detection marker found.")
    end
    println("Wrote $output")
    println("Wrote $sea_ice_output")
    nan_check_output !== nothing && println("Wrote $nan_check_output")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
