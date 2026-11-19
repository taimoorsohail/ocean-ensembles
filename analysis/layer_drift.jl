# Spin-up check: annual layer-mean Θ and S anomalies over the last N complete NE years vs the last N years of ACCESS-OM2-01.
# Meant to be `include`d in a persistent REPL; set N (default 10) before including to change the window.
using CairoMakie, JLD2, NCDatasets, Statistics, Printf

const LD_ROOT = "/data/gpfs/projects/punim2499/taimoor/gadi-ensembles"
const LD_PP = joinpath(LD_ROOT, "postprocessed", "RYF_control")
const LD_OM2 = joinpath(LD_ROOT, "ACCESS-OM2-01_ryf9091")
N = @isdefined(N) ? N : 10

coords = load(joinpath(LD_ROOT, "postprocessed", "grid_coords.jld2"))
H = -coords["bottom"]; area = coords["area"] .* (H .> 0.5)
layers = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, Inf), (0, Inf)]
vol(D1, D2) = sum(area .* (clamp.(H, D1, D2) .- D1))

# NE: complete years (≥ 70 five-day records) from the annual column-integral maps
complete(f) = jldopen(fh -> length(fh["T_integral_0_1000m_times"]) ≥ 70, f)
files = filter(complete, sort(readdir(joinpath(LD_PP, "annual"), join=true)))[end-N+1:end]
ne_years = [parse(Int, match(r"year_(\d+)", f).captures[1]) for f in files]
function ne_series(c)
    rows = map(files) do f
        d = jldopen(fh -> Dict(k => Float64.(fh[k][:, 1:size(H, 2)]) for k in keys(fh) if startswith(k, "$(c)_integral") && !endswith(k, "_times")), f)
        C(D) = D == 0 ? zeros(size(H)) : D == Inf ? d["$(c)_integral_0_1000m"] .+ d["$(c)_integral_1000m_bottom"] : d[@sprintf("%s_integral_0_%04dm", c, D)]
        [sum(area .* (C(D2) .- C(D1))) / vol(D1, D2) for (D1, D2) in layers]
    end
    [[r[k] for r in rows] for k in eachindex(layers)]
end

# OM2: yearly layer means, last N years. The stored whole-ocean means are Float32 (steps of 3.8e-6 near 35), so the
# whole ocean is rebuilt as the volume-weighted mean of the five layers.
om2_vol = NCDataset(joinpath(LD_OM2, "grid_ocean.nc")) do ds
    ht = Float64.(replace(ds["ht"][:, :], missing => 0.0)); A = Float64.(replace(ds["area_t"][:, :], missing => 0.0)) .* (ht .> 0)
    [sum(A .* (clamp.(ht, D1, D2) .- D1)) for (D1, D2) in layers[1:5]]
end
pats =[["0_100", "0-100", "0to100"], ["100_200", "100-200"], ["200_500", "200-500"], ["500_1000", "500-1000"],
        ["1000_b", "1000-b", "1000_bottom", "deep"], ["full", "total", "all", "whole", "0_b"]]
function om2_series(file, var)
    NCDataset(joinpath(LD_OM2, file)) do ds
        ks = collect(keys(ds))
        s = map(pats) do ps
            k = something(findfirst(k -> any(p -> occursin(var * "_" * p, k), ps), ks), findfirst(k -> any(p -> occursin(p, k), ps), ks), 0)
            k == 0 && return fill(NaN, N)
            y = Float64.(replace(ds[ks[k]][:], missing => NaN))
            var == "temp" && y[1] > 100 && (y .-= 273.15)
            var == "salt" && (y .*= 35.16504 / 35)
            y[end-N+1:end]
        end
        whole = sum(om2_vol .* s[1:5]) / sum(om2_vol)
        @printf("OM2 %s: rebuilt whole-ocean mean − stored = %.2g (max over years)\n", var, maximum(abs.(whole .- s[6])))
        s[6] = whole
        s
    end
end

trend(y) = ([ones(N) 0:N-1] \ y)[2]
names = ["0–100 m", "100–200 m", "200–500 m", "500–1000 m", "1000 m–bottom", "whole ocean"]
fig = Figure(size=(1300, 1500), fontsize=14)
for (col, c, var, file, name, unit, scale) in ((1, "T", "temp", "ts_layer_mean_temp_yearly_2130-2179.nc", "Θ", "K", 1.0),
                                               (2, "S", "salt", "ts_layer_mean_salt_yearly_2170-2179.nc", "S", "g kg⁻¹", 1.0))
    ne, om2 = ne_series(c), om2_series(file, var)
    for (row, lab) in enumerate(names)
        ax = Axis(fig[row, col]; title="$name, $lab", ylabel=unit, xlabel=row == length(names) ? "year of record" : "",
                  xticks=0:N-1)
        hlines!(ax, [0]; color=:gray70, linewidth=1)
        a, b = ne[row], om2[row]
        scatterlines!(ax, 0:N-1, scale .* (a .- a[1]); color="#1f8f4e", linewidth=2.5, markersize=7,
                      label="NumericalEarth ⅙°, years $(first(ne_years))–$(last(ne_years))")
        scatterlines!(ax, 0:N-1, scale .* (b .- b[1]); color="#6a3d9a", linewidth=2.5, markersize=7,
                      label="ACCESS-OM2-01 RYF9091, $(2180 - N)–2179")
        row == 1 && col == 1 && Legend(fig[length(names) + 1, 1:2], ax; orientation=:horizontal, framevisible=false, labelsize=15, tellwidth=false)
    end
end
Label(fig[0, 1:2], "Spin-up drift: annual layer means relative to the first year of each record (last $N years)"; fontsize=18, tellwidth=false)
out = joinpath(LD_ROOT, "figures", "RYF_control", "compare", "drift_timeseries_last$(N)yr.png")
save(out, fig; px_per_unit=2)
println("SAVED ", out)
