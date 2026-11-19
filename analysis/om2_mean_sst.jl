# ACCESS-OM2-01 RYF9091 mean SST (top level of the 2170–2179 mean temperature). `include` in the persistent REPL.
using CairoMakie, NCDatasets, Statistics

let path = "/data/gpfs/projects/punim2499/taimoor/gadi-ensembles/ACCESS-OM2-01_ryf9091/ocean_3d_mean_2170-2179.nc"
    sst, lon, lat = NCDataset(path) do ds
        v = ds["temp"]
        dn = dimnames(v)
        x = Float32.(replace(v[:, :, 1], missing => NaN))
        x, Float64.(ds[dn[1]][:]), Float64.(ds[dn[2]][:])
    end
    nanmax = maximum(filter(isfinite, sst))
    nanmax > 100 && (sst .-= 273.15f0)
    ok = filter(isfinite, sst)
    @info "OM2 mean SST" size(sst) extrema(ok)

    fig = Figure(size=(1500, 820), fontsize=16)
    ax = Axis(fig[1, 1]; title="ACCESS-OM2-01 RYF9091 mean SST, 2170–2179", xlabel="longitude (nominal)", ylabel="latitude (nominal)",
              aspect=DataAspect())
    hm = heatmap!(ax, lon, lat, sst; colormap=:thermal, colorrange=(-2, 31), nan_color=:gray80)
    Colorbar(fig[1, 2], hm; label="°C")
    xlims!(ax, extrema(lon)...); ylims!(ax, -80, 90)
    out = "/data/gpfs/projects/punim2499/taimoor/gadi-ensembles/figures/RYF_control/compare/OM2_mean_sst_2170-2179.png"
    save(out, fig; px_per_unit=2)
    println("SAVED ", out)
end
