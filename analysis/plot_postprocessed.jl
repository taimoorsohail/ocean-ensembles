# Figures from the post-processed set (postprocess_chunk.jl), so they keep working after raw output moves to Mediaflux.
# Usage: julia --project=<experiments/submission_scripts> plot_postprocessed.jl [run_name=RYF_control]
# Writes to figures/<run_name>/: time series, Hovmöllers (0.25° zonal means) and annual-mean maps of the latest full year.

using CairoMakie
using JLD2
using Printf
using Statistics

const RUN = length(ARGS) ≥ 1 ? ARGS[1] : "RYF_control"
const ROOT = "/data/gpfs/projects/punim2499/taimoor/gadi-ensembles"
const PP = joinpath(ROOT, "postprocessed", RUN)
const FIGDIR = joinpath(ROOT, "figures", RUN)
const NAN_PLOT_COLOR = :lightgray
const ρ₀ = 1026.0
const cₚ = 3991.86795711963
const DEPTHS = [0, 30, 100, 200, 500, 1000]
const LAYERS = ["0_0010m", "0_0030m", "0_0100m", "0_0200m", "0_0500m", "0_1000m", "1000m_bottom"]
const LAYER_TITLES = ["0–10 m", "0–30 m", "0–100 m", "0–200 m", "0–500 m", "0–1000 m", "1000 m–bottom"]
mkpath(FIGDIR)

dtag(d) = lpad(string(d), 4, '0') * "m"
years(t) = t ./ (365 * 86400)

coords = jldopen(fh -> Dict(k => fh[k] for k in keys(fh)), joinpath(ROOT, "postprocessed", "grid_coords.jld2"))
area, bottom = coords["area"], coords["bottom"]
area_ocean = sum(area[bottom .< 0])

#####
##### Reading, with later runs winning where restarts overlap in time
#####

run_number(file) = (m = match(r"_run(\d+)", file); isnothing(m) ? 0 : parse(Int, m.captures[1]))
part_number(file) = (m = match(r"_part(\d+)", file); isnothing(m) ? 0 : parse(Int, m.captures[1]))
ordered_files(kind) = sort(filter(f -> endswith(f, ".jld2"), readdir(joinpath(PP, kind))); by = f -> (run_number(f), part_number(f)))

# (times, values) of `var` across files; `getcol(fh, n)` extracts record n
function merged(kind, var; getcol)
    best = Dict{Float64, Tuple{Int, Any}}()
    for file in ordered_files(kind)
        jldopen(joinpath(PP, kind, file)) do fh
            haskey(fh, var) || return
            t = fh["t"]
            A = fh[var]
            for n in eachindex(t)
                t[n] > 0 || continue
                best[t[n]] = (run_number(file), getcol(A, n))   # files are in run order, so later runs overwrite
            end
        end
    end
    ts = sort(collect(keys(best)))
    return ts, [best[t][2] for t in ts]
end

series(var) = (r = merged("timeseries", var; getcol = (A, n) -> A[n]); (r[1], Float64.(r[2])))
function hovmoller_data(var)
    t, cols = merged("zonal", var; getcol = (A, n) -> A[:, n])
    lat = jldopen(fh -> fh["lat"], joinpath(PP, "zonal", first(ordered_files("zonal"))))
    return t, lat, isempty(cols) ? zeros(0, length(lat)) : permutedims(reduce(hcat, cols))
end

#####
##### Time series
#####

t, V = series("volume"); _, ∫T = series("T_integral"); _, ∫S = series("S_integral")
tN, eN = series("extent_north"); _, eS = series("extent_south")

panels = [("Global sea level from volume (mm)", t, (V .- V[1]) ./ area_ocean .* 1000),
          ("Mean Θ (°C)", t, ∫T ./ V), ("Mean S (g/kg)", t, ∫S ./ V),
          ("OHC anomaly (J)", t, ρ₀ * cₚ .* (∫T .- ∫T[1])), ("OSC anomaly (kg)", t, ρ₀ / 1000 .* (∫S .- ∫S[1])),
          ("Area-mean SST (°C)", series("T_0000m_areamean")...), ("Area-mean SSS (g/kg)", series("S_0000m_areamean")...),
          ("Area-mean SSH (m)", series("ssh_areamean")...), ("Area-mean MLD (m)", series("mld_areamean")...),
          ("Area-mean net heat flux, positive up (W/m²)", series("heat_flux_areamean")...),
          ("Area-mean net freshwater flux (m/s)", series("fw_flux_areamean")...)]
fig = Figure(size = (1300, 1300))
for (k, (title, x, y)) in enumerate(panels)
    ax = Axis(fig[cld(k, 3), mod1(k, 3)]; title, xlabel = "years")
    lines!(ax, years(x), y)
end
ax = Axis(fig[4, 3]; title = "Sea-ice extent (10⁶ km²)", xlabel = "years")
lines!(ax, years(tN), eN ./ 1e12, label = "north"); lines!(ax, years(tN), eS ./ 1e12, label = "south"); axislegend(ax)
save(joinpath(FIGDIR, "timeseries_global.png"), fig, px_per_unit = 2)
@info "saved timeseries_global"

fig = Figure(size = (900, 500)); ax = Axis(fig[1, 1]; title = "OHC anomaly by layer (J)", xlabel = "years")
for (L, title) in zip(LAYERS, LAYER_TITLES)
    tl, m = series("T_integral_$(L)_areamean")
    isempty(tl) && continue
    ohc = ρ₀ * cₚ .* m .* area_ocean
    lines!(ax, years(tl), ohc .- ohc[1], label = title)
end
axislegend(ax, position = :lt); save(joinpath(FIGDIR, "timeseries_OHC_layers.png"), fig, px_per_unit = 2)
@info "saved timeseries_OHC_layers"

#####
##### Hovmöllers (0.25° zonal means)
#####

function hovmoller!(fig, pos, title, var; scale = 1, colormap = :thermal, anomaly = false, label = "")
    t, lat, H = hovmoller_data(var)
    H = scale .* H
    nanmean(x) = (y = filter(isfinite, x); isempty(y) ? NaN : mean(y))
    anomaly && (H = H .- permutedims([nanmean(H[:, j]) for j in axes(H, 2)]))
    ax = Axis(fig[pos...]; title, xlabel = "years", ylabel = "latitude")
    x = filter(isfinite, vec(H))
    lim = anomaly ? (m = quantile(x, 0.99); (-m, m)) : (quantile(x, 0.01), quantile(x, 0.99))
    hm = heatmap!(ax, years(t), lat, H; colormap = anomaly ? :balance : colormap, colorrange = lim, nan_color = NAN_PLOT_COLOR)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label)
end

fig = Figure(size = (1500, 1400))
hovmoller!(fig, (1, 1), "Zonal-mean SST anomaly", "T_0000m"; anomaly = true, label = "°C")
hovmoller!(fig, (1, 3), "Zonal-mean SSS anomaly", "S_0000m"; anomaly = true, label = "g/kg")
hovmoller!(fig, (2, 1), "Zonal-mean SSH", "ssh"; colormap = :balance, label = "m")
hovmoller!(fig, (2, 3), "Zonal-mean MLD", "mld"; colormap = :deep, label = "m")
hovmoller!(fig, (3, 1), "Zonal-mean net heat flux, positive up", "heat_flux"; colormap = :balance, label = "W/m²")
hovmoller!(fig, (3, 3), "Zonal-mean OHC 0–1000 m anomaly", "T_integral_0_1000m"; scale = ρ₀ * cₚ, anomaly = true, label = "J/m²")
hovmoller!(fig, (4, 1), "Zonal-mean sea-ice concentration", "ice_concentration"; colormap = :ice, label = "")
hovmoller!(fig, (4, 3), "Zonal-mean Θ anomaly at 500 m", "T_0500m"; anomaly = true, label = "°C")
save(joinpath(FIGDIR, "hovmoller_zonal_means.png"), fig, px_per_unit = 2)
@info "saved hovmoller_zonal_means"

fig = Figure(size = (1300, 500))
for (col, var, label) in ((1, "T", "°C"), (3, "S", "g/kg"))
    cols = [series("$(var)_$(dtag(d))_areamean") for d in DEPTHS]
    n = minimum(length(c[2]) for c in cols)
    tt = cols[1][1][1:n]
    D = reduce(hcat, [c[2][1:n] for c in cols])
    D = D .- mean(D, dims = 1)
    m = maximum(abs, filter(isfinite, D))
    ax = Axis(fig[1, col]; title = "Area-mean $(var) anomaly by depth", xlabel = "years", ylabel = "depth (m)", yreversed = true)
    hm = heatmap!(ax, years(tt), Float64.(DEPTHS), D; colormap = :balance, colorrange = (-m, m))
    Colorbar(fig[1, col + 1], hm; label)
end
save(joinpath(FIGDIR, "hovmoller_depth_time.png"), fig, px_per_unit = 2)
@info "saved hovmoller_depth_time"

#####
##### Annual-mean maps of the latest full model year
#####

year_files = sort(filter(f -> startswith(f, "year_"), readdir(joinpath(PP, "annual"))))
nrecords(file) = jldopen(fh -> haskey(fh, "T_0000m_times") ? length(fh["T_0000m_times"]) : 0, joinpath(PP, "annual", file))
full = filter(f -> nrecords(f) ≥ 70, year_files)
if isempty(full)
    @warn "no complete model year yet; skipping annual maps"
else
    yfile = last(full)
    year = parse(Int, match(r"year_(-?\d+)", yfile).captures[1])
    Y = jldopen(fh -> Dict(k => fh[k] for k in keys(fh)), joinpath(PP, "annual", yfile))
    tag = @sprintf("year %d", year + 1)
    field(name) = Float64.(Y[name][:, 1:size(area, 2), 1])
    function masked(A, depth)
        B = copy(A); B[.!(bottom .< -max(depth, 0.5))] .= NaN; B
    end
    function robust(fields; symmetric = false)
        x = filter(isfinite, reduce(vcat, vec.(fields)))
        lo, hi = quantile(x, 0.01), quantile(x, 0.99)
        m = max(abs(lo), abs(hi))
        return symmetric ? (-m, m) : (lo, hi)
    end
    function map_panels(name, titles, fields; colormap = :thermal, symmetric = false, ncols = 3, label = "", shared = true)
        n = length(fields); nrows = cld(n, ncols)
        fig = Figure(size = ((shared ? 520 : 600)ncols, 320nrows))
        crange = robust(fields; symmetric); hm = nothing
        for (k, (title, A)) in enumerate(zip(titles, fields))
            col = shared ? mod1(k, ncols) : 2mod1(k, ncols) - 1
            ax = Axis(fig[cld(k, ncols), col]; title, aspect = DataAspect()); hidedecorations!(ax)
            hm = heatmap!(ax, A; colormap, colorrange = shared ? crange : robust([A]; symmetric), nan_color = NAN_PLOT_COLOR)
            shared || Colorbar(fig[cld(k, ncols), col + 1], hm)
        end
        shared && Colorbar(fig[1:nrows, ncols + 1], hm; label)
        save(joinpath(FIGDIR, name * ".png"), fig, px_per_unit = 2)
        @info "saved $name ($tag)"
    end

    titles = ["$d m, $tag" for d in DEPTHS]
    map_panels("map_T_slices_annual", titles, [masked(field("T_$(dtag(d))"), d) for d in DEPTHS]; label = "Θ (°C)")
    map_panels("map_S_slices_annual", titles, [masked(field("S_$(dtag(d))"), d) for d in DEPTHS]; colormap = :haline, label = "S (g/kg)")
    map_panels("map_speed_slices_annual", titles,
               [masked(sqrt.(field("u_$(dtag(d))") .^ 2 .+ field("v_$(dtag(d))") .^ 2), d) for d in DEPTHS];
               colormap = :speed, label = "speed of annual-mean flow (m/s)")
    for (var, label, cmap, sym) in (("ssh", "SSH (m)", :balance, true), ("heat_flux", "Net heat flux, positive up (W/m²)", :balance, true),
                                    ("fw_flux", "Net freshwater flux (m/s)", :balance, true), ("mld", "Mixed layer depth (m)", :deep, false))
        haskey(Y, var) && map_panels("map_$(var)_annual", ["$label, $tag"], [masked(field(var), 0)]; colormap = cmap, symmetric = sym, ncols = 1, label)
    end
    haskey(Y, "ice_concentration") && map_panels("map_ice_concentration_annual", ["ice concentration, $tag"],
                                                 [masked(field("ice_concentration"), 0)]; colormap = :ice, ncols = 1, label = "")
    layers = filter(L -> haskey(Y, "T_integral_$L"), LAYERS)
    lt = [LAYER_TITLES[findfirst(==(L), LAYERS)] for L in layers]
    map_panels("map_OHC_layers_annual", lt, [ρ₀ * cₚ .* masked(field("T_integral_$L"), 0) for L in layers]; ncols = 4, label = "OHC (J/m²)", shared = false)
    map_panels("map_OSC_layers_annual", lt, [ρ₀ / 1000 .* masked(field("S_integral_$L"), 0) for L in layers]; colormap = :haline, ncols = 4, label = "OSC (kg/m²)", shared = false)
end
