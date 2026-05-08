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
                             "worktmux_spartan-gpgpu011.log")
const DEFAULT_OUTPUT = joinpath(ROOT_DIR, "figures", "worktmux_spartan-gpgpu011_progress.png")
const DEFAULT_SEA_ICE_OUTPUT = joinpath(ROOT_DIR, "figures", "worktmux_spartan-gpgpu011_sea_ice.png")

const NUMBER = raw"(?:NaN|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"

function strip_terminal_codes(line)
    line = replace(line, r"\e\][^\a]*(?:\a|\e\\)" => "")
    line = replace(line, r"\e\[[0-?]*[ -/]*[@-~]" => "")
    return replace(line, '\r' => "")
end

parse_number(text) = text == "NaN" ? NaN : parse(Float64, text)

function pushvalue!(series, name, value)
    push!(get!(series, name, Float64[]), value)
end

function pad_missing!(series, names)
    for name in names
        pushvalue!(series, name, NaN)
    end
end

function parse_progress_log(path)
    series = Dict{String, Vector{Float64}}()

    main_re = Regex("time:\\s*($NUMBER) days, iteration:\\s*(\\d+).*?" *
                    "max\\|u\\|:\\s*\\(($NUMBER),\\s*($NUMBER),\\s*($NUMBER)\\).*?" *
                    "extrema\\(T\\):\\s*\\(($NUMBER),\\s*($NUMBER)\\).*?" *
                    "extrema\\(S\\):\\s*\\(($NUMBER),\\s*($NUMBER)\\).*?" *
                    "extrema\\(.\\):\\s*\\(($NUMBER),\\s*($NUMBER)\\).*?" *
                    "wall time:\\s*($NUMBER)\\s*(seconds|minutes)")
    wall_clock_re = Regex("Wall clock time:\\s*($NUMBER) days")
    sypd_re = Regex("SYPD:\\s*($NUMBER)")
    cfl_re = Regex("advective_cfl:\\s*($NUMBER)")

    optional_names = [
        "wall_clock_days",
        "SYPD",
        "advective_cfl",
        "sea_ice_h",
        "sea_ice_area",
        "sea_ice_qio",
        "sea_ice_Ssurf",
        "sea_ice_Sice",
        "sea_ice_Jsio",
        "sea_ice_Qio",
        "sea_ice_Qfrazil",
        "sea_ice_tauio_x",
        "sea_ice_tauio_y",
        "net_ocean_JT",
        "net_ocean_JS",
        "net_ocean_tau_x",
        "net_ocean_tau_y",
    ]
    records = 0

    for raw_line in eachline(path)
        line = strip_terminal_codes(raw_line)

        m = match(main_re, line)
        if m !== nothing
            records += 1
            pushvalue!(series, "time_days", parse_number(m.captures[1]))
            pushvalue!(series, "iteration", parse(Float64, m.captures[2]))
            pushvalue!(series, "max_abs_u", parse_number(m.captures[3]))
            pushvalue!(series, "max_abs_v", parse_number(m.captures[4]))
            pushvalue!(series, "max_abs_w", parse_number(m.captures[5]))
            pushvalue!(series, "T_max", parse_number(m.captures[6]))
            pushvalue!(series, "T_min", parse_number(m.captures[7]))
            pushvalue!(series, "S_max", parse_number(m.captures[8]))
            pushvalue!(series, "S_min", parse_number(m.captures[9]))
            pushvalue!(series, "eta_max", parse_number(m.captures[10]))
            pushvalue!(series, "eta_min", parse_number(m.captures[11]))

            wall_time = parse_number(m.captures[12])
            wall_units = m.captures[13]
            pushvalue!(series, "wall_time_seconds", wall_units == "minutes" ? 60 * wall_time : wall_time)
            pad_missing!(series, optional_names)
            continue
        end

        records == 0 && continue

        if (m = match(wall_clock_re, line)) !== nothing
            series["wall_clock_days"][end] = parse_number(m.captures[1])
        elseif (m = match(sypd_re, line)) !== nothing
            series["SYPD"][end] = parse_number(m.captures[1])
        elseif (m = match(cfl_re, line)) !== nothing
            series["advective_cfl"][end] = parse_number(m.captures[1])
        elseif occursin("sea ice at advective_cfl max:", line)
            values = [parse_number(m.match) for m in eachmatch(Regex(NUMBER), line)]
            length(values) < 14 && continue

            series["sea_ice_h"][end] = values[1]
            series["sea_ice_area"][end] = values[2]
            series["sea_ice_qio"][end] = values[3]
            series["sea_ice_Ssurf"][end] = values[4]
            series["sea_ice_Sice"][end] = values[5]
            series["sea_ice_Jsio"][end] = values[6]
            series["sea_ice_Qio"][end] = values[7]
            series["sea_ice_Qfrazil"][end] = values[8]
            series["sea_ice_tauio_x"][end] = values[9]
            series["sea_ice_tauio_y"][end] = values[10]
            series["net_ocean_JT"][end] = values[11]
            series["net_ocean_JS"][end] = values[12]
            series["net_ocean_tau_x"][end] = values[13]
            series["net_ocean_tau_y"][end] = values[14]
        end
    end

    return series
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
        ("T_max", "T max (degC)"),
        ("T_min", "T min (degC)"),
        ("S_max", "S max (g kg^-1)"),
        ("S_min", "S min (g kg^-1)"),
        ("eta_max", "eta max (m)"),
        ("eta_min", "eta min (m)"),
    ]

    return filter(p -> haskey(series, p[1]) && any(isfinite, series[p[1]]), panes)
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

    return filter(p -> haskey(series, p[1]) && any(isfinite, series[p[1]]), panes)
end

function finite_limits(x, y)
    mask = isfinite.(x) .& isfinite.(y)
    return x[mask], y[mask]
end

function plot_progress_makie(series; output_path=DEFAULT_OUTPUT, panes=progress_panes(series),
                             title="Progress parsed from $(basename(DEFAULT_LOG))")
    x = series["time_days"]
    isempty(panes) && error("No progress values found in log.")

    ncols = 2
    nrows = cld(length(panes), ncols)
    fig = Figure(size=(1500, 300nrows), fontsize=14)

    for (i, (name, label)) in enumerate(panes)
        row = cld(i, ncols)
        col = mod1(i, ncols)
        ax = Axis(fig[row, col], xlabel="model time (days)", ylabel=label, title=label)
        xf, yf = finite_limits(x, series[name])
        lines!(ax, xf, yf, color=:navy, linewidth=2)
        scatter!(ax, xf, yf, color=:navy, markersize=5)
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
                           title="Progress parsed from $(basename(DEFAULT_LOG))")
    x = series["time_days"]
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

            println(io, """<polyline points="$points" fill="none" stroke="#0f172a" stroke-width="2"/>""")
            println(io, """<text x="$(left + plot_w / 2)" y="$(top + plot_h + 42)" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#334155">model time (days)</text>""")
        end

        println(io, "</svg>")
    end

    return output_path
end

function plot_progress(series; output_path=DEFAULT_OUTPUT, panes=progress_panes(series),
                       title="Progress parsed from $(basename(DEFAULT_LOG))")
    if HAS_CAIROMAKIE
        return plot_progress_makie(series; output_path, panes, title)
    else
        return plot_progress_svg(series; output_path=replace(output_path, r"\.[^.]*$" => ".svg"), panes, title)
    end
end

function main()
    log_path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_LOG
    output_path = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_OUTPUT
    sea_ice_output_path = length(ARGS) >= 3 ? ARGS[3] : DEFAULT_SEA_ICE_OUTPUT

    series = parse_progress_log(log_path)
    output = plot_progress(series; output_path, panes=progress_panes(series),
                           title="Progress parsed from $(basename(log_path))")
    sea_ice_output = plot_progress(series; output_path=sea_ice_output_path, panes=sea_ice_panes(series),
                                   title="Sea ice diagnostics at advective CFL max")
    println("Parsed $(length(series["time_days"])) progress records.")
    println("Wrote $output")
    println("Wrote $sea_ice_output")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
