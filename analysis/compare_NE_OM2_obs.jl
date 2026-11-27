# Compare the NumericalEarth ⅙° RYF control (NE) with ACCESS-OM2-01 RYF9091 (OM2) and observations.
#
# Usage (from experiments/submission_scripts):
#   julia --project ../../analysis/compare_NE_OM2_obs.jl [seaice ssh gmsl mld sst sss fluxes speed spinup]   (default: all)
#
# NE: post-processed set in postprocessed/RYF_control (annual maps, 5-day zonal means and area means, monthly ice extent),
#     climatology over NE_YEARS (default 7 to the last complete year: T order 3, S order 1, no tracer biharmonic).
# OM2: native 0.1° exports of 01deg_jra55v13_ryf9091, model years 2170–2179, in OM2_DIR (pulled from Gadi v46).
# Observations, where an analogue exists: NSIDC Sea Ice Index v4 extent (1981–2010), AVISO MDT and monthly SLA (1993–2012),
#     Argo MLD climatology (Holte et al. 2017, density threshold), WOA23 1° decadal-average T/S.
# Spin-up quantities (global/layer Θ and S drift) are model-only: NE vs OM2.
#
# Maps are area-weighted averages onto a regular lon/lat grid (0.25° for display, 1° for differences against 1° observations).
# Model Θ / absolute salinity are compared with WOA in-situ T and practical salinity scaled by 35.16504/35 (as the model's ECCO
# initial condition); at the surface Θ ≈ T. OM2 practical salinity is scaled the same way.

using JLD2, NCDatasets, CairoMakie, Statistics, Printf, Dates

const ROOT = "/data/gpfs/projects/punim2499/taimoor/gadi-ensembles"
const PP = joinpath(ROOT, "postprocessed", "RYF_control")
const OBS = joinpath(ROOT, "obs")
const AVISO_DIR = get(ENV, "AVISO_DIR", "/data/gpfs/projects/punim2499/taimoor/ocean-ensembles/data/AVISO")
const OM2_DIR = get(ENV, "OM2_DIR", joinpath(ROOT, "ACCESS-OM2-01_ryf9091"))
const FIGDIR = joinpath(ROOT, "figures", "RYF_control", "compare")
const CACHE = joinpath(OBS, "cache")
mkpath(FIGDIR); mkpath(CACHE)

const YR = 365 * 86400.0
const MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
const MONTH_EDGES = cumsum([0; MONTH_DAYS])
const MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
const SA_PER_SP = 35.16504 / 35
const NE_CHANGE_DAY = 2400.0   # day the current setup started (T order 3, S order 1, no tracer biharmonic): drift year 0
const ρ₀ = 1026.0

const C_NE = "#1f8f4e"     # NumericalEarth: green
const C_OM2 = "#6a3d9a"    # ACCESS-OM2-01: purple
const C_OBS = "#262626"    # observations: near-black

const SECTIONS = isempty(ARGS) ? ["seaice", "ssh", "gmsl", "mld", "sst", "meanTS", "depth", "fluxes", "speed", "spinup"] : ARGS

nan(x) = x === missing ? NaN : Float64(x)
nanmean(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))
wrap180(λ) = mod(λ + 180, 360) - 180
have_om2(f) = isfile(joinpath(OM2_DIR, f)) || (@warn "OM2 file missing, skipping its curves" f; false)

#####
##### Regular-grid binning and helpers
#####

struct RegGrid
    res::Float64
    lon::Vector{Float64}
    lat::Vector{Float64}
end
RegGrid(res) = RegGrid(res, collect(-180 + res/2:res:180), collect(-90 + res/2:res:90))
const G25 = RegGrid(0.25)
const G1 = RegGrid(1.0)

"Area-weighted mean of `A` (any shape) onto `g`, binning by cell-centre lon/lat. Cells with non-finite A or zero weight are skipped."
function bin(g::RegGrid, lon, lat, w, A)
    Nx, Ny = length(g.lon), length(g.lat)
    num = zeros(Nx, Ny); den = zeros(Nx, Ny)
    @inbounds for I in eachindex(A)
        a = A[I]; (isfinite(a) && w[I] > 0 && isfinite(lon[I]) && isfinite(lat[I])) || continue
        i = clamp(floor(Int, (wrap180(lon[I]) + 180) / g.res) + 1, 1, Nx)
        j = clamp(floor(Int, (lat[I] + 90) / g.res) + 1, 1, Ny)
        num[i, j] += w[I] * a; den[i, j] += w[I]
    end
    M = [d > 0 ? n / d : NaN for (n, d) in zip(num, den)]
    # North of 65°N the tripolar grid cells are wider than fine bins, leaving empty rows: fill them from 1° bins
    jfold = findfirst(≥(65), g.lat)
    if g.res < 1 && !isnothing(jfold)
        C = bin(G1, lon, lat, w, A)
        r = round(Int, 1 / g.res)
        for j in jfold:Ny, i in 1:Nx
            isnan(M[i, j]) && (M[i, j] = C[cld(i, r), cld(j, r)])
        end
    end
    return M
end

coslat(g::RegGrid) = repeat(cosd.(g.lat)', length(g.lon), 1)
zonal(g::RegGrid, M) = [nanmean(view(M, :, j)) for j in axes(M, 2)]
function areamean(g::RegGrid, M; latmask = trues(length(g.lat)))
    w = coslat(g); s = 0.0; d = 0.0
    for I in CartesianIndices(M); (isfinite(M[I]) && latmask[I[2]]) || continue; s += w[I] * M[I]; d += w[I]; end
    return s / d
end
weighted_mean(A, w) = (m = isfinite.(A) .& (w .> 0); sum(A[m] .* w[m]) / sum(w[m]))

"model time (s, stamped at the end of the averaging window) → (model year, month 1–12) of the window centre"
function year_month(t, window)
    τ = t - window / 2
    y = floor(Int, τ / YR); d = mod(τ, YR) / 86400
    return y, clamp(searchsortedlast(MONTH_EDGES, d), 1, 12)
end

monthly_clim(ym, x, years) = [nanmean([x[n] for n in eachindex(x) if ym[n][2] == m && ym[n][1] in years]) for m in 1:12]
monthly_range(ym, x, years) = begin
    # spread of the monthly means across years
    per = [[nanmean([x[n] for n in eachindex(x) if ym[n] == (y, m)]) for y in years] for m in 1:12]
    ([minimum(filter(isfinite, p); init=NaN) for p in per], [maximum(filter(isfinite, p); init=NaN) for p in per])
end

#####
##### NE loaders
#####

const NE_COORDS = load(joinpath(ROOT, "postprocessed", "grid_coords.jld2"))
const NE_LON, NE_LAT, NE_AREA, NE_BOTTOM = NE_COORDS["lon"], NE_COORDS["lat"], NE_COORDS["area"], NE_COORDS["bottom"]
const NE_WET = NE_BOTTOM .< -0.5

function ne_complete_years()
    ys = Int[]
    for f in readdir(joinpath(PP, "annual"))
        y = parse(Int, match(r"year_(\d+)", f).captures[1])
        n = jldopen(fh -> (k = first(filter(k -> endswith(k, "_times"), collect(keys(fh)))); length(fh[k])), joinpath(PP, "annual", f))
        n ≥ 70 && push!(ys, y)
    end
    return sort(ys)
end
const NE_YEARS = let ys = ne_complete_years(); y0 = parse(Int, get(ENV, "NE_FIRST_YEAR", "7")); filter(≥(y0), ys) end
@info "NE climatology years" NE_YEARS

"mean of annual maps `name` over NE_YEARS, NaN on land"
function ne_annual(name)
    acc = zeros(size(NE_AREA)); n = 0
    for y in NE_YEARS
        f = joinpath(PP, "annual", @sprintf("year_%04d.jld2", y))
        A = jldopen(fh -> haskey(fh, name) ? Float64.(fh[name]) : nothing, f)
        isnothing(A) && continue
        acc .+= A[:, 1:size(NE_AREA, 2)]; n += 1
    end
    n == 0 && error("no annual maps of $name")
    M = acc ./ n; M[.!NE_WET] .= NaN
    return M
end

"concatenated 5-day (or monthly) area-mean series: (t, values) over all post-processed timeseries files with this prefix"
function ne_series(prefix, key)
    t = Float64[]; v = Float64[]
    for f in readdir(joinpath(PP, "timeseries"), join=true)
        startswith(basename(f), prefix) || continue
        jldopen(f) do fh
            haskey(fh, key) || return
            append!(t, fh["t"]); append!(v, fh[key])
        end
    end
    p = sortperm(t); t, v = t[p], v[p]
    keep = [true; diff(t) .> 0]
    return t[keep], v[keep]
end

"concatenated 5-day zonal means (lat × time)"
function ne_zonal(key)
    t = Float64[]; Z = Matrix{Float32}[]; lat = nothing
    for f in readdir(joinpath(PP, "zonal"), join=true)
        startswith(basename(f), "fields_5day") || continue
        jldopen(f) do fh
            haskey(fh, key) || return
            lat = fh["lat"]; append!(t, fh["t"]); push!(Z, fh[key])
        end
    end
    Zc = reduce(hcat, Z); p = sortperm(t); t, Zc = t[p], Zc[:, p]
    keep = [true; diff(t) .> 0]
    return lat, t[keep], Zc[:, keep]
end

#####
##### OM2 loaders (native 0.1° MOM5 / CICE5)
#####

om2_path(f) = joinpath(OM2_DIR, f)
om2_grid() = NCDataset(om2_path("grid_ocean.nc")) do ds
    (lon = nan.(ds["geolon_t"][:, :]), lat = nan.(ds["geolat_t"][:, :]), area = nan.(ds["area_t"][:, :]),
     lonu = haskey(ds, "geolon_c") ? nan.(ds["geolon_c"][:, :]) : nothing,
     latu = haskey(ds, "geolat_c") ? nan.(ds["geolat_c"][:, :]) : nothing,
     areau = haskey(ds, "area_u") ? nan.(ds["area_u"][:, :]) : nothing)
end
om2_icegrid() = NCDataset(om2_path("grid_ice.nc")) do ds
    (lon = nan.(ds["TLON"][:, :]), lat = nan.(ds["TLAT"][:, :]), area = nan.(ds["tarea"][:, :]))
end
om2_2d_mean(var::AbstractString) = NCDataset(ds -> (v = ds[var]; nan.(ndims(v) == 2 ? v[:, :] : v[:, :, 1])), om2_path("ocean_2d_mean_2170-2179.nc"))
om2_2d_mean(vars::Vector) = sum(om2_2d_mean(v) for v in vars)
function om2_2d_monthly_areamean(var, g)
    NCDataset(om2_path("ocean_2d_monthly_2170-2179.nc")) do ds
        [weighted_mean(nan.(ds[var][:, :, n]), g.area) for n in 1:size(ds[var], 3)]
    end
end
function om2_2d_monthly_zonal(var, g, rg::RegGrid)
    NCDataset(om2_path("ocean_2d_monthly_2170-2179.nc")) do ds
        reduce(hcat, [zonal(rg, bin(rg, g.lon, g.lat, g.area, nan.(ds[var][:, :, n]))) for n in 1:size(ds[var], 3)])
    end
end

"first variable in `ds` whose lowercase name contains any of `pats` (and is not a coordinate)"
function findvar(ds, pats)
    for k in keys(ds)
        lk = lowercase(k)
        any(p -> occursin(p, lk), pats) && ndims(ds[k]) ≥ 1 && !(lk in ("time", "time_bnds")) && return k
    end
    return nothing
end

#####
##### Observations
#####

"NSIDC Sea Ice Index v4 daily extent → 1981–2010 monthly median and 10–90 % range (10⁶ km²)"
function nsidc_monthly(hemi)
    f = joinpath(OBS, "nsidc", hemi == :N ? "N_seaice_extent_daily_v4.0.csv" : "S_seaice_extent_daily_v4.0.csv")
    vals = [Float64[] for _ in 1:12]
    for line in eachline(f)
        p = strip.(split(line, ','))
        y = tryparse(Int, p[1]); isnothing(y) && continue
        (1981 ≤ y ≤ 2010) || continue
        push!(vals[parse(Int, p[2])], parse(Float64, p[4]))
    end
    q(v, p) = quantile(v, p)
    return [median(v) for v in vals], [q(v, 0.1) for v in vals], [q(v, 0.9) for v in vals]
end

"AVISO: mean dynamic topography (ADT − SLA on 1993-01-01, i.e. the 1993–2012 MDT) on 0.25°, and monthly GMSL 1993–2012"
function aviso()
    cache = joinpath(CACHE, "aviso_mdt_gmsl.jld2")
    isfile(cache) && return load(cache, "mdt", "gmsl", "time")
    mdt, lon, lat = NCDataset(joinpath(AVISO_DIR, "adt_sla_AVISODaily_1993-01-01.nc")) do ds
        nan.(ds["adt"][:, :, 1]) .- nan.(ds["sla"][:, :, 1]), Float64.(ds["longitude"][:]), Float64.(ds["latitude"][:])
    end
    LON = repeat(lon, 1, length(lat)); LAT = repeat(lat', length(lon), 1); W = cosd.(LAT)
    mdt25 = bin(G25, LON, LAT, W, mdt)
    gmsl, time = NCDataset(joinpath(AVISO_DIR, "sla_AVISOMonthly_1993-2012.nc")) do ds
        [weighted_mean(nan.(ds["sla"][:, :, n]), W) for n in 1:size(ds["sla"], 3)], ds["time"][:]
    end
    jldsave(cache; mdt=mdt25, gmsl, time=Dates.value.(Date.(time) .- Date(1993, 1, 1)) ./ 365.25)
    return mdt25, gmsl, Dates.value.(Date.(time) .- Date(1993, 1, 1)) ./ 365.25
end

"Argo MLD monthly climatology (density threshold), 1° (lon −180..180)"
function argo_mld()
    NCDataset(joinpath(OBS, "argo_mld", "Argo_mixedlayers_monthlyclim_04142022.nc")) do ds
        M = nan.(ds["mld_dt_mean"][:, :, :])            # month × lon × lat
        lon = Float64.(ds["lon"][:]); lat = Float64.(ds["lat"][:])
        # put onto G1 ordering (lon −179.5..179.5, lat −89.5..89.5)
        LON = repeat(lon, 1, length(lat)); LAT = repeat(lat', length(lon), 1)
        [bin(G1, LON, LAT, ones(size(LON)), M[m, :, :]) for m in 1:12]
    end
end

"WOA23 annual decadal-average objectively analysed field at `depth` (m) on G1; salinity scaled to SA"
function woa(var, depth)
    f = joinpath(OBS, "woa23", var == :T ? "woa23_decav_t00_01.nc" : "woa23_decav_s00_01.nc")
    NCDataset(f) do ds
        k = argmin(abs.(ds["depth"][:] .- depth))
        A = nan.(ds[var == :T ? "t_an" : "s_an"][:, :, k, 1])
        var == :S && (A .*= SA_PER_SP)
        lon = Float64.(ds["lon"][:]); lat = Float64.(ds["lat"][:])
        bin(G1, repeat(lon, 1, length(lat)), repeat(lat', length(lon), 1), ones(length(lon), length(lat)), A)
    end
end

#####
##### Plot helpers
#####

function mapax(f, pos, title)
    ax = Axis(f[pos...]; title, aspect=DataAspect(), xticks=-180:60:180, yticks=-60:30:60, titlesize=14)
    hidedecorations!(ax; grid=false, ticklabels=false); return ax
end
function mapplot!(f, pos, g::RegGrid, M, title; colorrange, colormap=:thermal, label="")
    ax = mapax(f, pos, title)
    hm = heatmap!(ax, g.lon, g.lat, M; colorrange, colormap, nan_color=(:gray80), lowclip=Makie.to_colormap(colormap)[1], highclip=Makie.to_colormap(colormap)[end])
    xlims!(ax, -180, 180); ylims!(ax, -80, 90)
    return ax, hm
end
robust(Ms...; p=0.02) = (v = filter(isfinite, reduce(vcat, vec.(Ms))); (quantile(v, p), quantile(v, 1 - p)))
symrange(Ms...; p=0.98) = (v = filter(isfinite, abs.(reduce(vcat, vec.(Ms)))); m = quantile(v, p); (-m, m))
footer!(f, text) = Label(f[end+1, :], text; fontsize=11, color=:gray30, halign=:left, justification=:left, tellwidth=false)

ne_label() = "NumericalEarth ⅙° (years $(first(NE_YEARS))–$(last(NE_YEARS)))"
const OM2_LABEL = "ACCESS-OM2-01 RYF9091 (2170–2179)"

savefig(name, fig) = (p = joinpath(FIGDIR, name); save(p, fig; px_per_unit=1.5); println("saved ", p))

#####
##### 1. Sea-ice extent
#####

function fig_seaice()
    t, N = ne_series("sea_ice_monthly", "extent_north"); _, S = ne_series("sea_ice_monthly", "extent_south")
    scale = maximum(filter(isfinite, N)) > 1e9 ? 1e-12 : 1.0          # m² → 10⁶ km²
    N .*= scale; S .*= scale
    ym = [year_month(τ, YR / 12) for τ in t]

    om2 = nothing
    if have_om2("ts_seaice_extent_monthly.nc")
        om2 = NCDataset(om2_path("ts_seaice_extent_monthly.nc")) do ds
            kn = findvar(ds, ["north", "nh", "arctic"]); ks = findvar(ds, ["south", "sh", "antarctic"])
            n = nan.(ds[kn][:]); s = nan.(ds[ks][:])
            sc = maximum(filter(isfinite, n)) > 1e9 ? 1e-12 : 1.0
            (n = n .* sc, s = s .* sc)
        end
    end

    fig = Figure(size=(1400, 560), fontsize=15)
    for (col, hemi, x, title) in ((1, :N, N, "Arctic"), (2, :S, S, "Antarctic"))
        ax = Axis(fig[1, col]; title="$title sea-ice extent: seasonal cycle", ylabel="10⁶ km²", xticks=(1:12, MONTHS))
        med, lo, hi = nsidc_monthly(hemi)
        band!(ax, 1:12, lo, hi; color=(:gray, 0.25)); lines!(ax, 1:12, med; color=C_OBS, linewidth=3, label="NSIDC SII v4, 1981–2010 median (10–90 %)")
        lo_ne, hi_ne = monthly_range(ym, x, NE_YEARS)
        band!(ax, 1:12, lo_ne, hi_ne; color=(C_NE, 0.25)); lines!(ax, 1:12, monthly_clim(ym, x, NE_YEARS); color=C_NE, linewidth=3, label=ne_label())
        if !isnothing(om2)
            o = hemi == :N ? om2.n : om2.s
            last120 = o[end-119:end]; Mo = reshape(last120, 12, 10)
            band!(ax, 1:12, vec(minimum(Mo, dims=2)), vec(maximum(Mo, dims=2)); color=(C_OM2, 0.25))
            lines!(ax, 1:12, vec(mean(Mo, dims=2)); color=C_OM2, linewidth=3, label=OM2_LABEL)
        end
        col == 1 && axislegend(ax; position=:lb, framevisible=false, labelsize=11)

    end
    footer!(fig, "Extent = area with concentration ≥ 15 %. Shading: range of monthly means across the climatology years (models), 10th–90th percentile of daily values (NSIDC).")
    savefig("seaice_extent.png", fig)
end

#####
##### 2. SSH: mean dynamic topography
#####

function fig_ssh()
    ne = bin(G25, NE_LON, NE_LAT, NE_AREA .* NE_WET, ne_annual("ssh")); ne .-= areamean(G25, ne)
    mdt, _, _ = aviso(); mdt .-= areamean(G25, mdt)
    om2 = nothing
    if have_om2("ocean_2d_mean_2170-2179.nc")
        g = om2_grid(); om2 = bin(G25, g.lon, g.lat, g.area, om2_2d_mean("sea_level")); om2 .-= areamean(G25, om2)
    end
    fig = Figure(size=(1900, 950), fontsize=15)
    cr = (-1.8, 1.2); crd = (-0.5, 0.5)
    _, hm = mapplot!(fig, (1, 1), G25, ne, ne_label(); colorrange=cr, colormap=:balance)
    isnothing(om2) || mapplot!(fig, (1, 2), G25, om2, OM2_LABEL; colorrange=cr, colormap=:balance)
    mapplot!(fig, (1, 3), G25, mdt, "AVISO MDT (1993–2012)"; colorrange=cr, colormap=:balance)
    Colorbar(fig[1, 4], hm; label="time-mean SSH − global mean (m)")
    rms(d) = sqrt(nanmean(d .^ 2))
    if !isnothing(om2)
        d = ne .- om2
        mapplot!(fig, (2, 2), G25, d, @sprintf("NE − ACCESS-OM2-01 (RMS %.2f m)", rms(d)); colorrange=crd, colormap=Reverse(:RdBu))
    end
    d = ne .- mdt
    _, hmd = mapplot!(fig, (2, 3), G25, d, @sprintf("NE − AVISO (RMS %.2f m)", rms(d)); colorrange=crd, colormap=Reverse(:RdBu))
    Colorbar(fig[2, 4], hmd; label="difference (m)")
    footer!(fig, "Each field has its own area mean removed. AVISO MDT = ADT − SLA (CMEMS DUACS, 1993–2012 reference).")
    savefig("ssh_mean.png", fig)
end

#####
##### 3. Global-mean sea level, monthly
#####

function fig_gmsl()
    t, η = ne_series("fields_5day", "ssh_areamean")
    ym = [year_month(τ, 5 * 86400.0) for τ in t]
    keys_ = sort(unique(ym)); ne_m = [nanmean(η[ym .== Ref(k)]) for k in keys_]
    ne_t = [k[1] + (k[2] - 0.5) / 12 for k in keys_]
    _, ag, at = aviso()
    om2 = have_om2("ocean_2d_monthly_2170-2179.nc") ? om2_2d_monthly_areamean("sea_level", om2_grid()) : nothing

    fig = Figure(size=(1300, 900), fontsize=14)
    ax = Axis(fig[1, 1:2]; title="Global-mean sea level, monthly means", xlabel="model year (NE)", ylabel="GMSL − mean (mm)")
    lines!(ax, ne_t, 1e3 .* (ne_m .- mean(ne_m)); color=C_NE, linewidth=2, label="NumericalEarth ⅙°, years $(floor(Int, first(ne_t)))–$(floor(Int, last(ne_t)))")
    axislegend(ax; position=:lt, framevisible=false)

    ax2 = Axis(fig[2, 1]; title="ACCESS-OM2-01 (2170–2179) and AVISO (1993–2012)", xlabel="years from start of record", ylabel="mm")
    lines!(ax2, at, 1e3 .* (ag .- mean(ag)); color=C_OBS, linewidth=2, label="AVISO (incl. ~3 mm/yr rise)")
    isnothing(om2) || lines!(ax2, ((1:length(om2)) .- 0.5) ./ 12, 1e3 .* (om2 .- mean(om2)); color=C_OM2, linewidth=2, label="ACCESS-OM2-01")
    axislegend(ax2; position=:lt, framevisible=false)

    ax3 = Axis(fig[2, 2]; title="Seasonal cycle (linear trend removed)", xticks=(1:12, MONTHS), ylabel="mm")
    detr(x, tt) = (X = [ones(length(tt)) tt]; x .- X * (X \ x))
    seas(x, months) = [mean(x[months .== m]) for m in 1:12]
    selne = [k[1] in NE_YEARS for k in keys_]
    lines!(ax3, 1:12, 1e3 .* seas(detr(ne_m[selne], ne_t[selne]), last.(keys_)[selne]); color=C_NE, linewidth=3, label="NE")
    lines!(ax3, 1:12, 1e3 .* seas(detr(ag, at), [month(Date(1993, 1, 15) + Month(n - 1)) for n in 1:length(ag)]); color=C_OBS, linewidth=3, label="AVISO")
    isnothing(om2) || lines!(ax3, 1:12, 1e3 .* seas(detr(om2, collect(1.0:length(om2))), [mod1(n, 12) for n in 1:length(om2)]); color=C_OM2, linewidth=3, label="OM2")
    axislegend(ax3; position=:lt, framevisible=false)
    footer!(fig, "Model GMSL is the area mean of the free surface (Boussinesq, no steric expansion term), so it reflects the net freshwater input; AVISO includes steric and mass change.")
    savefig("gmsl_monthly.png", fig)
end

#####
##### 4. Mixed-layer depth
#####

"mean over NE_YEARS of the seasonal (quarterly) map `name`_Q`q`; nothing if not yet built"
function ne_seasonal(name, q)
    acc = zeros(size(NE_AREA)); n = 0
    for y in NE_YEARS
        f = joinpath(PP, "seasonal", @sprintf("year_%04d.jld2", y)); isfile(f) || continue
        A = jldopen(fh -> haskey(fh, "$(name)_Q$q") ? Float64.(fh["$(name)_Q$q"]) : nothing, f)
        isnothing(A) && continue
        acc .+= A[:, 1:size(NE_AREA, 2)]; n += 1
    end
    n == 0 && return nothing
    M = acc ./ n; M[.!NE_WET] .= NaN
    return M
end

"hemispheric season composite on a regular grid: NH from `nh`, SH from `sh`"
hemi(g::RegGrid, nh, sh) = [g.lat[j] ≥ 0 ? nh[i, j] : sh[i, j] for i in eachindex(g.lon), j in eachindex(g.lat)]

"WOA23 seasonal (q = 1 JFM … 4 OND) decadal-average field at `depth` on G1; salinity scaled to SA"
function woa_season(var, q; depth = 0)
    f = joinpath(OBS, "woa23", @sprintf("woa23_decav_%s%d_01.nc", var == :T ? "t" : "s", 12 + q))
    NCDataset(f) do ds
        k = argmin(abs.(ds["depth"][:] .- depth))
        A = nan.(ds[var == :T ? "t_an" : "s_an"][:, :, k, 1]); var == :S && (A .*= SA_PER_SP)
        lon = Float64.(ds["lon"][:]); lat = Float64.(ds["lat"][:])
        bin(G1, repeat(lon, 1, length(lat)), repeat(lat', length(lon), 1), ones(length(lon), length(lat)), A)
    end
end

"""
Summer / winter maps (hemispheric composites: summer = NH JAS + SH JFM, winter = NH JFM + SH JAS), one figure per model:
rows summer and winter, columns model | observations | model − observations.
"""
function fig_seasonal(; ne_name, om2_var, om2_scale = 1.0, obs_q, obs_label, tag, quantity, unit, crs, crds, cmap, note)
    obs = Dict(:summer => hemi(G1, obs_q(3), obs_q(1)), :winter => hemi(G1, obs_q(1), obs_q(3)))
    models = Pair{String, Any}[]
    q1, q3 = ne_seasonal(ne_name, 1), ne_seasonal(ne_name, 3)
    if !isnothing(q1) && !isnothing(q3)
        b(g, M) = bin(g, NE_LON, NE_LAT, NE_AREA .* NE_WET, M)
        push!(models, "NE" => (label = ne_label(), Q = Dict(1 => (b(G25, q1), b(G1, q1)), 3 => (b(G25, q3), b(G1, q3)))))
    else
        @warn "NE seasonal maps of $ne_name not built yet (seasonal backfill)"
    end
    if have_om2("ocean_2d_monthly_2170-2179.nc")
        g = om2_grid()
        mq = NCDataset(om2_path("ocean_2d_monthly_2170-2179.nc")) do ds
            nt = size(ds[om2_var], 3)
            Dict(q => om2_scale .* mean([nan.(ds[om2_var][:, :, n]) for n in 1:nt if cld(mod1(n, 12), 3) == q]) for q in (1, 3))
        end
        o(gr, M) = bin(gr, g.lon, g.lat, g.area, M)
        push!(models, "OM2" => (label = OM2_LABEL, Q = Dict(q => (o(G25, mq[q]), o(G1, mq[q])) for q in (1, 3))))
    end
    for (name, m) in models
        fig = Figure(size=(1900, 900), fontsize=15)
        for (row, season, title, nhq, shq) in ((1, :summer, "summer (NH JAS, SH JFM)", 3, 1), (2, :winter, "winter (NH JFM, SH JAS)", 1, 3))
            M25 = hemi(G25, m.Q[nhq][1], m.Q[shq][1]); M1 = hemi(G1, m.Q[nhq][2], m.Q[shq][2])
            _, hm = mapplot!(fig, (row, 1), G25, M25, "$(m.label): $title"; colorrange=crs[row], colormap=cmap)
            mapplot!(fig, (row, 2), G1, obs[season], "$obs_label: $title"; colorrange=crs[row], colormap=cmap)
            Colorbar(fig[row, 3], hm; label="$quantity ($unit)")
            d = M1 .- obs[season]
            _, hmd = mapplot!(fig, (row, 4), G1, d, @sprintf("%s − obs (mean %+.2f, RMS %.2f)", name, areamean(G1, d), sqrt(nanmean(d .^ 2))); colorrange=crds[row], colormap=:balance)
            Colorbar(fig[row, 5], hmd; label="Δ $quantity ($unit)")
        end
        footer!(fig, note * " Model maps shown at 0.25°, differences at 1°.")
        savefig("$(tag)_seasonal_$(name).png", fig)
    end
end

function fig_mld()
    argo = argo_mld()
    argo_q(q) = [nanmean([argo[m][I] for m in 3q-2:3q]) for I in CartesianIndices(argo[1])]
    fig_seasonal(ne_name="mld", om2_var="mld", obs_q=argo_q, obs_label="Argo (Holte et al.)", tag="mld", quantity="MLD", unit="m",
                 crs=((0, 120), (0, 500)), crds=((-60, 60), (-250, 250)), cmap=:deep,
                 note="Argo: monthly mean MLD from the 0.03 kg m⁻³ density threshold (Holte et al. 2017, 2022 update), 1°. Model MLD uses a buoyancy criterion; criteria differ slightly.")
end

function fig_sst_sss()
    fig_seasonal(ne_name="T_0000m", om2_var="surface_temp", obs_q=q -> woa_season(:T, q), obs_label="WOA23", tag="sst", quantity="SST", unit="°C",
                 crs=((-2, 30), (-2, 30)), crds=((-4, 4), (-4, 4)), cmap=:thermal,
                 note="WOA23 decadal-average seasonal climatology (1°), in-situ surface T compared with model surface Θ.")
    fig_seasonal(ne_name="S_0000m", om2_var="surface_salt", om2_scale=SA_PER_SP, obs_q=q -> woa_season(:S, q), obs_label="WOA23", tag="sss", quantity="SSS", unit="g kg⁻¹",
                 crs=((32, 38), (32, 38)), crds=((-1.5, 1.5), (-1.5, 1.5)), cmap=:haline,
                 note="WOA23 practical salinity and OM2 salt scaled by 35.16504/35 to absolute salinity.")
end

"""
Global volume-mean Θ and S anomalies for the drift comparison: NE from the setup change (year 0 = day NE_CHANGE_DAY, 5-day means)
and the last N years of ACCESS-OM2-01 (N = length of the NE record, yearly means), each relative to its own year 0.
"""
function drift_series()
    t = Float64[]; Ti = Float64[]; Si = Float64[]; V = Float64[]
    for f in readdir(joinpath(PP, "timeseries"), join=true)
        b = basename(f); (startswith(b, "global_integrals") || startswith(b, "integrals_5day")) || continue
        d = load(f); append!(t, d["t"]); append!(Ti, d["T_integral"]); append!(Si, d["S_integral"]); append!(V, d["volume"])
    end
    p = sortperm(t); t, Ti, Si, V = t[p], Ti[p], Si[p], V[p]; k = [true; diff(t) .> 0]; t, Ti, Si, V = t[k], Ti[k], Si[k], V[k]
    sel = t .>= NE_CHANGE_DAY * 86400
    x = (t[sel] .- NE_CHANGE_DAY * 86400) ./ YR
    Θ = (Ti ./ V)[sel]; S = (Si ./ V)[sel]
    N = floor(Int, last(x))                              # complete years since the change
    yr = floor.(Int, x)
    Θa = [mean(Θ[yr .== k]) for k in 0:N-1]; Sa_ = [mean(S[yr .== k]) for k in 0:N-1]
    ne = (x = collect(0:N-1), Θ = Θa .- Θa[1], S = Sa_ .- Sa_[1])
    om2 = nothing
    sdir = joinpath(OM2_DIR, "ocean_scalar")
    yearly = joinpath(OM2_DIR, "ts_global_mean_temp_salt_yearly.nc")   # 1900–2179, yearly means
    if isfile(yearly)
        # The monthly ocean_scalar files only cover 10 years, so longer NE records use the yearly file
        NCDataset(yearly) do ds
            T = nan.(ds["temp_global_ave"][:]); Sa = nan.(ds["salt_global_ave"][:]) .* SA_PER_SP; y = ds["year"][:]
            r = length(T)-N+1:length(T)
            om2 = (x = collect(0:N-1), Θ = T[r] .- T[r[1]], S = Sa[r] .- Sa[r[1]], years = (y[r[1]], y[r[end]]))
        end
    elseif isdir(sdir)
        files = sort(filter(isfile, [joinpath(sdir, d, "ocean", "ocean_scalar.nc") for d in readdir(sdir)]))
        T = Float64[]; Sa = Float64[]; tm = Any[]
        for f in files
            NCDataset(f) do ds
                append!(T, vec(nan.(ds["temp_global_ave"][:, :]))); append!(Sa, vec(nan.(ds["salt_global_ave"][:, :])) .* SA_PER_SP); append!(tm, ds["time"][:])
            end
        end
        r = length(T)-12N+1:length(T)                    # last N years of monthly means → N annual means
        Ty = [mean(T[r][12k+1:12k+12]) for k in 0:N-1]; Sy = [mean(Sa[r][12k+1:12k+12]) for k in 0:N-1]
        om2 = (x = collect(0:N-1), Θ = Ty .- Ty[1], S = Sy .- Sy[1], years = (Dates.year(tm[r[1]]), Dates.year(tm[r[end]])))
    else
        @warn "OM2 monthly ocean_scalar files missing" sdir
    end
    return ne, om2, N
end

function drift_panels!(fig, row, ne, om2)
    for (col, key, name, unit) in ((1, :Θ, "Global volume-mean Θ anomaly", "K"), (2, :S, "Global volume-mean S anomaly", "g kg⁻¹"))
        ax = Axis(fig[row, col]; title="$name (annual mean) relative to year 0", xlabel="year", ylabel=unit)
        hlines!(ax, [0]; color=:gray60, linewidth=1)
        scatterlines!(ax, ne.x, getproperty(ne, key); color=C_NE, linewidth=2.5, markersize=10, label="NumericalEarth ⅙° (year 0 = first year after setup change)")
        isnothing(om2) || scatterlines!(ax, om2.x, getproperty(om2, key); color=C_OM2, linewidth=2.5, markersize=10, marker=:rect,
                                        label="ACCESS-OM2-01 RYF9091 $(om2.years[1])–$(om2.years[2])")
        ax.xticks = 0:max(1, cld(length(ne.x), 10)):length(ne.x)-1; xlims!(ax, -0.3, length(ne.x) - 0.7)
        col == 1 && axislegend(ax; position=:lt, framevisible=false, labelsize=12)
    end
end

"Global volume-mean Θ and S drift: NE since the setup change vs the last equally long stretch of ACCESS-OM2-01"
function fig_meanTS()
    ne, om2, N = drift_series()
    fig = Figure(size=(1500, 600), fontsize=15)
    drift_panels!(fig, 1, ne, om2)
    footer!(fig, "Annual means, anomalies relative to each record's year 0. NE: years since the setup change (day 2400). ACCESS-OM2-01: its last $N years, as the late-spin-up target.")
    savefig("mean_T_S_timeseries.png", fig)
end

#####
##### 5–6. Surface (and subsurface) temperature and salinity vs WOA23
#####

const OM2_3D_DEPTHS = Ref{Union{Nothing, Vector{Float64}}}(nothing)
"OM2 10-year-mean 3D field at the level nearest `depth` (or surface field at 0 m)"
function om2_tracer(var, depth)
    if depth == 0
        A = om2_2d_mean(var == :T ? "surface_temp" : "surface_salt")
    else
        A = NCDataset(om2_path("ocean_3d_mean_2170-2179.nc")) do ds
            z = Float64.(ds[findvar(ds, ["st_ocean"])][:]); k = argmin(abs.(z .- depth))
            nan.(ds[var == :T ? "temp" : "salt"][:, :, k])
        end
    end
    var == :S && (A .*= SA_PER_SP)
    return A
end

function fig_tracer(var; depths = [0, 100, 500, 1000])
    name = var == :T ? "Θ" : "S"; unit = var == :T ? "°C" : "g kg⁻¹"
    g = have_om2("ocean_2d_mean_2170-2179.nc") ? om2_grid() : nothing
    for d in depths
        tag = lpad(string(d), 4, '0') * "m"
        ne = ne_annual("$(var)_$tag")
        wet = NE_BOTTOM .< -max(d, 0.5); ne[.!wet] .= NaN
        ne1 = bin(G1, NE_LON, NE_LAT, NE_AREA .* wet, ne)
        obs = woa(var, d)
        om1 = nothing
        if !isnothing(g) && (d == 0 || isfile(om2_path("ocean_3d_mean_2170-2179.nc")))
            om1 = bin(G1, g.lon, g.lat, g.area, om2_tracer(var, d))
        end
        fig = Figure(size=(1500, 1000), fontsize=14)
        cr = robust(obs); crd = var == :T ? (-4, 4) : (-1, 1)
        _, hm = mapplot!(fig, (1, 1), G1, obs, "WOA23 (1° decadal average), $d m"; colorrange=cr, colormap=var == :T ? :thermal : :haline)
        Colorbar(fig[1, 2], hm; label="$name ($unit)")
        b1 = ne1 .- obs
        ax = Axis(fig[1, 3]; title="Zonal-mean difference from WOA23, $d m", xlabel="latitude", ylabel=unit)
        hlines!(ax, [0]; color=C_OBS, linewidth=1.5)
        lines!(ax, G1.lat, zonal(G1, b1); color=C_NE, linewidth=2, label="NumericalEarth ⅙°")
        isnothing(om1) || lines!(ax, G1.lat, zonal(G1, om1 .- obs); color=C_OM2, linewidth=2, label="ACCESS-OM2-01")
        xlims!(ax, -80, 90); axislegend(ax; position=:lb, framevisible=false)
        _, hmd = mapplot!(fig, (2, 1), G1, b1, @sprintf("NumericalEarth − WOA23 (RMS %.2f, mean %+.2f)", sqrt(nanmean(b1 .^ 2)), areamean(G1, b1)); colorrange=crd, colormap=:balance)
        if !isnothing(om1)
            b2 = om1 .- obs
            mapplot!(fig, (2, 3), G1, b2, @sprintf("ACCESS-OM2-01 − WOA23 (RMS %.2f, mean %+.2f)", sqrt(nanmean(b2 .^ 2)), areamean(G1, b2)); colorrange=crd, colormap=:balance)
        end
        Colorbar(fig[2, 2], hmd; label="bias ($unit)")
        footer!(fig, "Models binned to 1°. $(ne_label()); $OM2_LABEL. WOA salinity (practical) and ACCESS-OM2-01 salt scaled by 35.16504/35 to absolute salinity; WOA in-situ T compared with model Θ.")
        savefig("$(var == :T ? "temperature" : "salinity")_$(tag).png", fig)
    end
end

#####
##### 7. Surface fluxes and speed (model vs model)
#####

function fig_model_pair(ne_name, om2_var, title, unit, file; ne_scale=1.0, om2_scale=1.0, cr=nothing, cmap=:balance)
    ne = ne_scale .* bin(G25, NE_LON, NE_LAT, NE_AREA .* NE_WET, ne_annual(ne_name))
    om2 = nothing
    if have_om2("ocean_2d_mean_2170-2179.nc")
        g = om2_grid()
        lon, lat, area = om2_var == "surface_speed" && !isnothing(g.lonu) ? (g.lonu, g.latu, g.areau) : (g.lon, g.lat, g.area)
        om2 = om2_scale .* bin(G25, lon, lat, area, om2_2d_mean(om2_var))
    end
    cr = something(cr, symrange(ne))
    fig = Figure(size=(1500, 900), fontsize=14)
    _, hm = mapplot!(fig, (1, 1), G25, ne, @sprintf("%s (global mean %.3g)", ne_label(), areamean(G25, ne)); colorrange=cr, colormap=cmap)
    isnothing(om2) || mapplot!(fig, (1, 2), G25, om2, @sprintf("%s (global mean %.3g)", OM2_LABEL, areamean(G25, om2)); colorrange=cr, colormap=cmap)
    Colorbar(fig[1, 3], hm; label="$title ($unit)")
    ax = Axis(fig[2, 1:2]; title="Zonal mean", xlabel="latitude", ylabel=unit)
    lines!(ax, G25.lat, zonal(G25, ne); color=C_NE, linewidth=2, label="NE")
    isnothing(om2) || lines!(ax, G25.lat, zonal(G25, om2); color=C_OM2, linewidth=2, label="OM2")
    hlines!(ax, [0]; color=:gray60); xlims!(ax, -80, 90); axislegend(ax; position=:lt, framevisible=false)
    return fig
end

function fig_fluxes()
    f = fig_model_pair("heat_flux", ["sfc_hflux_coupler", "frazil_3d_int_z"], "Net surface heat flux into the ocean", "W m⁻²", ""; ne_scale=-1.0, cr=(-150, 150))
    footer!(f, "NE heat_flux (net incl. penetrating shortwave, positive up) sign-flipped. OM2: sfc_hflux_coupler + frazil_3d_int_z (positive down); the advective heat of P−E and runoff (relative to 0 °C) is left out, as NE carries none.")
    savefig("heat_flux.png", f)
    f = fig_model_pair("fw_flux", "pme_river", "Net freshwater flux into the ocean", "kg m⁻² s⁻¹ × 10⁵", ""; ne_scale=-1e5, om2_scale=1e5, cr=(-10, 10))
    footer!(f, "NE: net_ocean_freshwater_flux (from the surface salinity flux, reference S = 35; positive up) sign-flipped. OM2: pme_river (P − E + runoff, positive into the ocean).")
    savefig("freshwater_flux.png", f)
end

function fig_speed()
    u = ne_annual("u_0000m"); v = ne_annual("v_0000m")
    sp = sqrt.(u .^ 2 .+ v .^ 2)
    # write a temporary map name through the same path as the other pairs
    ne = bin(G25, NE_LON, NE_LAT, NE_AREA .* NE_WET, sp)
    om2 = nothing
    if have_om2("ocean_2d_mean_2170-2179.nc")
        g = om2_grid(); lon, lat, area = isnothing(g.lonu) ? (g.lon, g.lat, g.area) : (g.lonu, g.latu, g.areau)
        om2 = bin(G25, lon, lat, area, om2_2d_mean("surface_speed"))
    end
    fig = Figure(size=(1500, 900), fontsize=14)
    cr = (0, 0.8)
    _, hm = mapplot!(fig, (1, 1), G25, ne, ne_label(); colorrange=cr, colormap=:speed)
    isnothing(om2) || mapplot!(fig, (1, 2), G25, om2, OM2_LABEL; colorrange=cr, colormap=:speed)
    Colorbar(fig[1, 3], hm; label="surface speed (m s⁻¹)")
    ax = Axis(fig[2, 1:2]; title="Zonal mean", xlabel="latitude", ylabel="m s⁻¹")
    lines!(ax, G25.lat, zonal(G25, ne); color=C_NE, linewidth=2, label="NE")
    isnothing(om2) || lines!(ax, G25.lat, zonal(G25, om2); color=C_OM2, linewidth=2, label="OM2")
    xlims!(ax, -80, 90); axislegend(ax; position=:lt, framevisible=false)
    footer!(fig, "NE: speed of the annual-mean surface velocity (from 5-day means of u, v); OM2: 10-year mean of the speed of monthly-mean k = 1 velocity. Both understate eddy speeds.")
    savefig("surface_speed.png", fig)
end

#####
##### 8. Spin-up drift (model vs model)
#####

"Layer-mean Θ and S drift profiles (linear trends): NE over NE_YEARS vs the last N years of ACCESS-OM2-01 (model-only)"
function fig_spinup()
    N = haskey(ENV, "DRIFT_YEARS") ? parse(Int, ENV["DRIFT_YEARS"]) : drift_series()[3]
    layers = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, Inf), (0, Inf)]
    H = -NE_BOTTOM; area = NE_AREA .* NE_WET
    vol(D1, D2) = sum(area .* (clamp.(H, D1, D2) .- D1))
    function ne_trends(c)
        Cint(d, D) = D == 0 ? zeros(size(H)) : D == Inf ? Float64.(d["$(c)_integral_0_1000m"]) .+ Float64.(d["$(c)_integral_1000m_bottom"]) : Float64.(d[@sprintf("%s_integral_0_%04dm", c, D)])
        rows = Dict{Int, Vector{Float64}}()
        for y in NE_YEARS
            d = jldopen(fh -> Dict(k => fh[k] for k in keys(fh) if startswith(k, "$(c)_integral") && !endswith(k, "_times")), joinpath(PP, "annual", @sprintf("year_%04d.jld2", y)))
            rows[y] = [sum(area .* (Cint(d, D2)[:, 1:size(H, 2)] .- Cint(d, D1)[:, 1:size(H, 2)])) / vol(D1, D2) for (D1, D2) in layers]
        end
        X = [ones(length(NE_YEARS)) NE_YEARS]
        [(X \ [rows[y][k] for y in NE_YEARS])[2] for k in eachindex(layers)]
    end
    pats = [["0_100", "0-100", "0to100"], ["100_200", "100-200"], ["200_500", "200-500"], ["500_1000", "500-1000"], ["1000_b", "1000-b", "1000_bottom", "deep"], ["full", "total", "all", "whole", "0_b"]]
    function om2_trends(file, var)
        have_om2(file) || return nothing
        NCDataset(om2_path(file)) do ds
            map(pats) do ps
                k = findvar(ds, [var * "_" * p for p in ps]); k = something(k, findvar(ds, ps)); isnothing(k) && return NaN
                y = nan.(ds[k][:]); var == "temp" && y[1] > 100 && (y .-= 273.15); var == "salt" && (y .*= SA_PER_SP)
                y = y[max(1, end-N+1):end]; Xo = [ones(length(y)) 1:length(y)]; (Xo \ y)[2]
            end
        end
    end
    zbot = 5000.0
    stairs(tr) = (xs = Float64[]; zs = Float64[];
                  for (k, (z1, z2)) in enumerate(layers[1:5]); push!(xs, tr[k], tr[k]); push!(zs, z1, isinf(z2) ? zbot : z2); end; (xs, zs))
    fig = Figure(size=(1200, 750), fontsize=15)
    for (col, c, var, file, name, unit) in ((1, "T", "temp", "ts_layer_mean_temp_yearly_2130-2179.nc", "Θ", "K yr⁻¹"),
                                            (2, "S", "salt", "ts_layer_mean_salt_yearly_2170-2179.nc", "S", "g kg⁻¹ yr⁻¹"))
        ne_tr = ne_trends(c); om2_tr = om2_trends(file, var)
        ax = Axis(fig[1, col]; title="Layer-mean $name drift", xlabel=unit, ylabel=col == 1 ? "depth (m)" : "", yreversed=true,
                  yscale=sqrt, yticks=[0, 100, 200, 500, 1000, 2000, 5000], ytickformat=v -> [x == 5000 ? "bottom" : string(round(Int, x)) for x in v])
        vlines!(ax, [0]; color=:gray60, linewidth=1)
        xs, zs = stairs(ne_tr); lines!(ax, xs, zs; color=C_NE, linewidth=3, label="NumericalEarth ⅙°, years $(first(NE_YEARS))–$(last(NE_YEARS))")
        vlines!(ax, [ne_tr[end]]; color=C_NE, linestyle=:dash, linewidth=1.5, label=@sprintf("NE whole ocean %.2g", ne_tr[end]))
        if !isnothing(om2_tr)
            xs, zs = stairs(om2_tr); lines!(ax, xs, zs; color=C_OM2, linewidth=3, label="ACCESS-OM2-01, last $N years")
            vlines!(ax, [om2_tr[end]]; color=C_OM2, linestyle=:dash, linewidth=1.5, label=@sprintf("OM2 whole ocean %.2g", om2_tr[end]))
        end
        ylims!(ax, zbot, 0)
        axislegend(ax; position=:rb, framevisible=false, labelsize=11)
    end
    footer!(fig, "Linear trends of annual layer means (0–100, 100–200, 200–500, 500–1000 m, 1000 m–bottom); dashed: whole ocean. No observational analogue. NE: current setup (since day 2400); OM2: its last $N years (late spin-up target).")
    savefig("layer_drift.png", fig)
end

#####
##### Run
#####

for s in SECTIONS
    try
        s == "seaice" && fig_seaice()
        s == "ssh" && fig_ssh()
        s == "gmsl" && fig_gmsl()
        s == "mld" && fig_mld()
        s == "sst" && fig_sst_sss()
        s == "depth" && (fig_tracer(:T); fig_tracer(:S))
        s == "meanTS" && fig_meanTS()
        s == "fluxes" && fig_fluxes()
        s == "speed" && fig_speed()
        s == "spinup" && fig_spinup()
    catch err
        @error "Section failed" s exception=(err, catch_backtrace())
    end
end
