# Figures for the output suite of github.com/taimoorsohail/ocean-ensembles/discussions/103 (items 1–6):
# 2D maps (time mean and latest record), global time series, and Hovmöllers.
# Usage: julia --project=<experiments/submission_scripts> plot_output_suite.jl [first_run]
# Follows the conventions of analysis2d_horizontal.jl (index-space heatmaps, land in light grey) and
# analysis_mht.jl (Hovmöller as heatmap of time × latitude).

using CairoMakie
using JLD2
using Logging
using Printf
using Statistics
using Oceananigans

const OUTPUT_PATH = expanduser(get(ENV, "OUTPUT_PATH", "/data/gpfs/projects/punim2499/taimoor/gadi-ensembles/outputs/"))
const FIGDIR = expanduser(get(ENV, "FIGDIR", "/data/gpfs/projects/punim2499/taimoor/gadi-ensembles/figures/output_suite/"))
const RESOLUTION = "sxtdeg"
const FIRST_RUN = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 23   # first run written with the #103 suite
const NAN_PLOT_COLOR = :lightgray
const ρ₀ = 1026.0          # kg m⁻³ (multiply ∫Θ by ρ₀ cₚ for heat, ∫S by ρ₀/1000 for salt)
const cₚ = 3991.86795711963
const SLICE_LEVELS = [75, 67, 56, 48, 36, 26]
const SLICE_DEPTHS = [0, 30, 100, 200, 500, 1000]
const LAYERS = ["0_0010m", "0_0030m", "0_0100m", "0_0200m", "0_0500m", "0_1000m", "1000m_bottom"]
const LAYER_TITLES = ["0–10 m", "0–30 m", "0–100 m", "0–200 m", "0–500 m", "0–1000 m", "1000 m–bottom"]

mkpath(FIGDIR)

#####
##### Reading
#####

run_files(stem) = sort(filter(f -> occursin(Regex("^$(stem)_$(RESOLUTION)_RYF_run(\\d+)\\.jld2\$"), f) &&
                                   parse(Int, match(r"run(\d+)", f).captures[1]) ≥ FIRST_RUN,
                              readdir(OUTPUT_PATH)))

record_keys(fh) = sort(parse.(Int, filter(k -> k != "serialized", collect(keys(fh["timeseries/t"])))))

# Concatenate one variable across runs, dropping records repeated at run boundaries
function load_series(stem, var)
    times = Float64[]; data = Any[]
    for file in run_files(stem)
        jldopen(joinpath(OUTPUT_PATH, file)) do fh
            (haskey(fh, "timeseries/t") && haskey(fh, "timeseries/$var")) || return
            for i in record_keys(fh)
                t = fh["timeseries/t/$i"]
                (!isempty(times) && t ≤ times[end]) && continue
                push!(times, t); push!(data, fh["timeseries/$var/$i"])
            end
        end
    end
    return times ./ 86400, data
end

scalar_series(stem, var) = (s = load_series(stem, var); (s[1], [Float64(x[1]) for x in s[2]]))
field2d(x) = Float64.(x[:, 1:1080, 1])   # v has an extra north face row

grid = with_logger(NullLogger()) do
    jldopen(joinpath(OUTPUT_PATH, first(run_files("global_surface"))))["serialized/grid"]
end
ug = grid.underlying_grid
interior2d(A) = (P = parent(A); ndims(P) == 3 ? P[ug.Hx+1:ug.Hx+ug.Nx, ug.Hy+1:ug.Hy+ug.Ny, 1] : P[ug.Hx+1:ug.Hx+ug.Nx, ug.Hy+1:ug.Hy+ug.Ny])
φ = Float64.(interior2d(ug.φᶜᶜᵃ))
area = Float64.(interior2d(ug.Azᶜᶜᵃ))
bottom = Float64.(interior2d(grid.immersed_boundary.bottom_height))
ocean = bottom .< 0
area_ocean = sum(area[ocean])

mask(A) = (B = copy(A); B[.!ocean] .= NaN; B)
area_mean(A) = (m = ocean .& isfinite.(A); sum(A[m] .* area[m]) / sum(area[m]))
years(days) = days ./ 365

#####
##### 2D maps
#####

function robust_range(fields; symmetric = false)
    x = filter(isfinite, reduce(vcat, vec.(fields)))
    lo, hi = quantile(x, 0.01), quantile(x, 0.99)
    symmetric && (m = max(abs(lo), abs(hi)); return (-m, m))
    return (lo, hi)
end

function map_panels(name, titles, fields; colormap = :thermal, symmetric = false, ncols = 3, label = "", shared = true)
    n = length(fields); nrows = cld(n, ncols)
    fig = Figure(size = ((shared ? 520 : 600)ncols, 320nrows))
    crange = robust_range(fields; symmetric)
    hm = nothing
    for (k, (title, A)) in enumerate(zip(titles, fields))
        col = shared ? mod1(k, ncols) : 2mod1(k, ncols) - 1
        ax = Axis(fig[cld(k, ncols), col]; title, aspect = DataAspect())
        hidedecorations!(ax)
        hm = heatmap!(ax, mask(A); colormap, colorrange = shared ? crange : robust_range([A]; symmetric), nan_color = NAN_PLOT_COLOR)
        shared || Colorbar(fig[cld(k, ncols), col + 1], hm)
    end
    shared && Colorbar(fig[1:nrows, ncols + 1], hm; label)
    save(joinpath(FIGDIR, name * ".png"), fig, px_per_unit = 2)
    @info "saved $name"
end

time_mean(data) = mean(field2d.(data))

# Item 1: slices of T, S and speed at each depth (time mean)
for (var, cmap, label) in (("T", :thermal, "Θ (°C)"), ("S", :haline, "S (g/kg)"))
    means = [time_mean(load_series("global_$(lev)_fields", var)[2]) for lev in SLICE_LEVELS]
    map_panels("map_$(var)_slices_timemean", ["$(d) m" for d in SLICE_DEPTHS], means; colormap = cmap, label)
end
speeds = map(SLICE_LEVELS) do lev
    _, u = load_series("global_$(lev)_fields", "u"); _, v = load_series("global_$(lev)_fields", "v")
    mean(sqrt.(field2d(a) .^ 2 .+ field2d(b) .^ 2) for (a, b) in zip(u, v))
end
map_panels("map_speed_slices_timemean", ["$(d) m" for d in SLICE_DEPTHS], speeds; colormap = :speed, label = "|u| (m/s)")

# Items 2–3: surface fields (time mean and latest)
surface_vars = [("ssh", "SSH (m)", :balance, true), ("heat_flux", "Net heat flux, positive upward (W/m²)", :balance, true),
                ("fw_flux", "Net freshwater flux (m/s)", :balance, true), ("mld", "Mixed layer depth (m)", :deep, false)]
for (stat, f) in (("timemean", time_mean), ("latest", d -> field2d(last(d))))
    for (var, label, cmap, sym) in surface_vars
        map_panels("map_$(var)_$(stat)", [label], [f(load_series("global_surface", var)[2])];
                   colormap = cmap, symmetric = sym, ncols = 1, label)
    end
end

# Item 4: sea-ice concentration
_, ice = load_series("global_sea_ice", "ice_concentration")
map_panels("map_ice_concentration", ["time mean", "latest month"], [time_mean(ice), field2d(last(ice))];
           colormap = :ice, ncols = 2, label = "concentration")

# Item 6: heat and salt content per layer (time mean, J/m² and kg/m²)
layers = filter(L -> !isempty(load_series("global_column_integrals", "T_integral_$L")[1]), LAYERS)
titles = [LAYER_TITLES[findfirst(==(L), LAYERS)] for L in layers]
map_panels("map_OHC_layers_timemean", titles,
           [ρ₀ * cₚ .* time_mean(load_series("global_column_integrals", "T_integral_$L")[2]) for L in layers];
           ncols = 4, label = "OHC (J/m²)", shared = false)
map_panels("map_OSC_layers_timemean", titles,
           [ρ₀ / 1000 .* time_mean(load_series("global_column_integrals", "S_integral_$L")[2]) for L in layers];
           colormap = :haline, ncols = 4, label = "OSC (kg/m²)", shared = false)

#####
##### Time series
#####

t, V = scalar_series("global_integrals", "volume")
_, ∫T = scalar_series("global_integrals", "T_integral")
_, ∫S = scalar_series("global_integrals", "S_integral")
tN, eN = scalar_series("global_sea_ice", "extent_north")
_, eS = scalar_series("global_sea_ice", "extent_south")

surface_mean(var) = (s = load_series("global_surface", var); (s[1], [area_mean(field2d(x)) for x in s[2]]))
ts, ssh = surface_mean("ssh"); _, Q = surface_mean("heat_flux"); _, F = surface_mean("fw_flux"); _, mld = surface_mean("mld")
tt, sst = (s = load_series("global_75_fields", "T"); (s[1], [area_mean(field2d(x)) for x in s[2]]))
_, sss = (s = load_series("global_75_fields", "S"); (s[1], [area_mean(field2d(x)) for x in s[2]]))

fig = Figure(size = (1300, 1300))
panels = [("Global sea level from volume (mm)", t, (V .- V[1]) ./ area_ocean .* 1000),
          ("Mean Θ (°C)", t, ∫T ./ V), ("Mean S (g/kg)", t, ∫S ./ V),
          ("OHC anomaly (J)", t, ρ₀ * cₚ .* (∫T .- ∫T[1])), ("OSC anomaly (kg)", t, ρ₀ / 1000 .* (∫S .- ∫S[1])),
          ("Area-mean SST (°C)", tt, sst), ("Area-mean SSS (g/kg)", tt, sss),
          ("Area-mean SSH (m)", ts, ssh), ("Area-mean MLD (m)", ts, mld),
          ("Area-mean net heat flux, positive up (W/m²)", ts, Q), ("Area-mean net freshwater flux (m/s)", ts, F)]
for (k, (title, x, y)) in enumerate(panels)
    ax = Axis(fig[cld(k, 3), mod1(k, 3)]; title, xlabel = "years")
    lines!(ax, years(x), y)
end
ax = Axis(fig[4, 3]; title = "Sea-ice extent (10⁶ km²)", xlabel = "years")
lines!(ax, years(tN), eN ./ 1e12, label = "north"); lines!(ax, years(tN), eS ./ 1e12, label = "south"); axislegend(ax)
save(joinpath(FIGDIR, "timeseries_global.png"), fig, px_per_unit = 2)
@info "saved timeseries_global"

# Layer heat content (area-integrated column integrals)
fig = Figure(size = (900, 500)); ax = Axis(fig[1, 1]; title = "OHC anomaly by layer (J)", xlabel = "years")
for (L, title) in zip(layers, titles)
    tl, data = load_series("global_column_integrals", "T_integral_$L")
    ohc = [ρ₀ * cₚ * sum(filter(isfinite, field2d(x)[ocean] .* area[ocean])) for x in data]
    lines!(ax, years(tl), ohc .- ohc[1], label = title)
end
axislegend(ax, position = :lt); save(joinpath(FIGDIR, "timeseries_OHC_layers.png"), fig, px_per_unit = 2)
@info "saved timeseries_OHC_layers"

#####
##### Hovmöllers
#####

const LAT_EDGES = -80:1.0:90
lat_centers = collect(LAT_EDGES[1:end-1] .+ 0.5)
lat_bin = clamp.(searchsortedlast.(Ref(LAT_EDGES), φ), 1, length(lat_centers))

function zonal_mean(A)
    num = zeros(length(lat_centers)); den = zeros(length(lat_centers))
    @inbounds for I in eachindex(A)
        (ocean[I] && isfinite(A[I])) || continue
        b = lat_bin[I]; num[b] += A[I] * area[I]; den[b] += area[I]
    end
    return [d > 0 ? n / d : NaN for (n, d) in zip(num, den)]
end

function hovmoller!(fig, pos, title, days, data; transform = identity, colormap = :thermal, anomaly = false, label = "")
    H = reduce(hcat, [zonal_mean(transform(field2d(x))) for x in data])'   # time × latitude
    anomaly && (H = H .- mean(H, dims = 1))
    ax = Axis(fig[pos...]; title, xlabel = "years", ylabel = "latitude")
    lim = anomaly ? (m = quantile(filter(isfinite, vec(H)), 0.99); (-m, m)) : robust_range([H])
    hm = heatmap!(ax, years(days), lat_centers, H; colormap = anomaly ? :balance : colormap, colorrange = lim, nan_color = NAN_PLOT_COLOR)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label)
end

fig = Figure(size = (1500, 1400))
hovmoller!(fig, (1, 1), "Zonal-mean SST anomaly", load_series("global_75_fields", "T")...; anomaly = true, label = "°C")
hovmoller!(fig, (1, 3), "Zonal-mean SSS anomaly", load_series("global_75_fields", "S")...; anomaly = true, label = "g/kg")
hovmoller!(fig, (2, 1), "Zonal-mean SSH", load_series("global_surface", "ssh")...; colormap = :balance, label = "m")
hovmoller!(fig, (2, 3), "Zonal-mean MLD", load_series("global_surface", "mld")...; colormap = :deep, label = "m")
hovmoller!(fig, (3, 1), "Zonal-mean net heat flux", load_series("global_surface", "heat_flux")...; colormap = :balance, label = "W/m²")
hovmoller!(fig, (3, 3), "Zonal-mean OHC 0–1000 m anomaly", load_series("global_column_integrals", "T_integral_0_1000m")...;
           transform = A -> ρ₀ * cₚ .* A, anomaly = true, label = "J/m²")
hovmoller!(fig, (4, 1), "Zonal-mean sea-ice concentration", load_series("global_sea_ice", "ice_concentration")...;
           colormap = :ice, label = "")
save(joinpath(FIGDIR, "hovmoller_zonal_means.png"), fig, px_per_unit = 2)
@info "saved hovmoller_zonal_means"

# Depth × time: area-mean Θ and S anomaly on the six slice depths
fig = Figure(size = (1300, 500))
for (col, var, label) in ((1, "T", "°C"), (3, "S", "g/kg"))
    t1 = load_series("global_75_fields", var)[1]
    D = reduce(hcat, [[area_mean(field2d(x)) for x in load_series("global_$(lev)_fields", var)[2]] for lev in SLICE_LEVELS])
    D = D .- mean(D, dims = 1)
    m = maximum(abs, filter(isfinite, D))
    ax = Axis(fig[1, col]; title = "Area-mean $(var) anomaly by depth", xlabel = "years", ylabel = "depth (m)", yreversed = true)
    hm = heatmap!(ax, years(t1), Float64.(SLICE_DEPTHS), D; colormap = :balance, colorrange = (-m, m))
    Colorbar(fig[1, col + 1], hm; label)
end
save(joinpath(FIGDIR, "hovmoller_depth_time.png"), fig, px_per_unit = 2)
@info "saved hovmoller_depth_time"
