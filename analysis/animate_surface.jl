# Animations of NumericalEarth ⅙° surface fields over one or more years of raw output.
#
# Usage (from experiments/submission_scripts):
#   julia --project ../../analysis/animate_surface.jl <fields_5day_….jld2 …> <sea_ice_monthly_….jld2 …>   (any number, in time order)
#
# 5-day means of SST, SSS, surface speed, SSH, net heat and freshwater flux, MLD and top-level w are binned area-weighted
# onto a 0.25° lon/lat grid (north of 55°N empty bins of the tripolar grid are filled from 1° bins); monthly sea-ice
# concentration goes onto Arctic and Antarctic polar (azimuthal equidistant) grids. Videos go to figures/RYF_control/animations/.

using JLD2, CairoMakie, Printf, Statistics

const ROOT = "/data/gpfs/projects/punim2499/taimoor/gadi-ensembles"
const OUTDIR = joinpath(ROOT, "figures", "RYF_control", "animations")
mkpath(OUTDIR)
const YR = 365 * 86400.0
const MONTH_EDGES = cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

const C = load(joinpath(ROOT, "postprocessed", "grid_coords.jld2"))
const LON, LAT, AREA = C["lon"], C["lat"], C["area"]
const NX, NY = size(AREA)
const WET = C["bottom"] .< -0.5
wrap180(λ) = mod(λ + 180, 360) - 180

#####
##### Binning with precomputed bin indices
#####

struct Binner
    idx::Vector{Int}      # target bin of each source cell (0 = skip)
    w::Vector{Float64}
    n::Int
end
function Binner(xbin, ybin, nx, ny, keep)
    idx = zeros(Int, NX * NY); w = zeros(NX * NY)
    for I in 1:NX*NY
        keep[I] && WET[I] && AREA[I] > 0 || continue
        i, j = xbin(I), ybin(I)
        (1 ≤ i ≤ nx && 1 ≤ j ≤ ny) || continue
        idx[I] = i + (j - 1) * nx; w[I] = AREA[I]
    end
    return Binner(idx, w, nx * ny)
end
function apply(b::Binner, A)
    num = zeros(b.n); den = zeros(b.n)
    @inbounds for I in eachindex(b.idx)
        k = b.idx[I]; a = A[I]
        (k > 0 && isfinite(a)) || continue
        num[k] += b.w[I] * a; den[k] += b.w[I]
    end
    return [d > 0 ? n / d : NaN for (n, d) in zip(num, den)]
end

const RES = 0.25
const GLON = collect(-180 + RES/2:RES:180); const GLAT = collect(-80 + RES/2:RES:90)
const B25 = Binner(I -> floor(Int, (wrap180(LON[I]) + 180) / RES) + 1, I -> floor(Int, (LAT[I] + 80) / RES) + 1,
                   length(GLON), length(GLAT), trues(NX * NY))
const B1 = Binner(I -> floor(Int, (wrap180(LON[I]) + 180)) + 1, I -> floor(Int, LAT[I] + 80) + 1, 360, 170, trues(NX * NY))
const B2 = Binner(I -> floor(Int, (wrap180(LON[I]) + 180) / 2) + 1, I -> floor(Int, (LAT[I] + 80) / 2) + 1, 180, 85, trues(NX * NY))
const JFOLD = findfirst(≥(55), GLAT)

function regrid(A)
    M = reshape(apply(B25, A), length(GLON), length(GLAT))
    C1 = reshape(apply(B1, A), 360, 170)
    r = round(Int, 1 / RES)
    for j in JFOLD:length(GLAT), i in eachindex(GLON)
        isnan(M[i, j]) && (M[i, j] = C1[cld(i, r), cld(j, r)])
    end
    C2 = reshape(apply(B2, A), 180, 85)   # cells near the pole are wider than 1°
    for j in findfirst(≥(80), GLAT):length(GLAT), i in eachindex(GLON)
        isnan(M[i, j]) && (M[i, j] = C2[cld(i, 2r), cld(j, 2r)])
    end
    return M
end

# polar grids: r = 90 − |lat| (degrees from the pole), out to 40° (50° latitude)
const PRES = 0.2; const PR = 40.0
const PX = collect(-PR + PRES/2:PRES:PR); const NP = length(PX)
polar(hemi) = Binner(I -> floor(Int, ((90 - hemi * LAT[I]) * sind(LON[I]) + PR) / PRES) + 1,
                     I -> floor(Int, (-hemi * (90 - hemi * LAT[I]) * cosd(LON[I]) + PR) / PRES) + 1,
                     NP, NP, vec(hemi .* LAT .> 45))
const BN, BS = polar(1), polar(-1)

#####
##### Reading
#####

frames(f, v) = sort(filter(k -> all(isdigit, k), collect(keys(f["timeseries/$v"]))), by = k -> parse(Int, k))
read2d(f, v, k) = Float64.(f["timeseries/$v/$k"][:, 1:NY, 1])
function datelabel(t, window)
    τ = t - window / 2; y = floor(Int, τ / YR); d = mod(τ, YR) / 86400
    m = clamp(searchsortedlast(MONTH_EDGES, d), 1, 12)
    return @sprintf("model year %d  ·  day %3d (%s)", y, floor(Int, d) + 1, MONTH_NAMES[m])
end

"surface velocity at cell centres from face velocities, in grid directions"
function centred_velocity(f, k)
    u = Float64.(f["timeseries/u_0000m/$k"][:, :, 1]); v = Float64.(f["timeseries/v_0000m/$k"][:, :, 1])
    return (u[:, 1:NY] .+ circshift(u[:, 1:NY], (-1, 0))) ./ 2, (v[:, 1:NY] .+ v[:, 2:NY+1]) ./ 2
end
"surface speed (rotation-invariant, so fine on the tripolar grid)"
speed(f, k) = let (uc, vc) = centred_velocity(f, k); sqrt.(uc .^ 2 .+ vc .^ 2) end
"surface eddy kinetic energy about the time mean of the animated window (UBAR, VBAR, set in animate_fields)"
const UBAR = Ref(zeros(NX, NY)); const VBAR = Ref(zeros(NX, NY))
eke(f, k) = let (uc, vc) = centred_velocity(f, k); 0.5 .* ((uc .- UBAR[]) .^ 2 .+ (vc .- VBAR[]) .^ 2) end
"SSH minus its area mean"
ssh_anom(f, k) = (η = read2d(f, "ssh", k); m = WET .& isfinite.(η); η .- sum(η[m] .* AREA[m]) / sum(AREA[m]))

#                name               reader                                              title                                      unit                cr            cmap
const FIELDS = [
    ("sst",            (f, k) -> read2d(f, "T_0000m", k),                      "Sea surface temperature",                     "Θ (°C)",             (-2, 31),     :thermal),
    ("sss",            (f, k) -> read2d(f, "S_0000m", k),                      "Sea surface salinity",                        "S (g kg⁻¹)",         (31, 38),     :haline),
    ("surface_speed",  speed,                                                   "Surface current speed",                       "m s⁻¹",              (0, 1.2),     :speed),
    ("ssh",            ssh_anom,                                                "Sea surface height (global mean removed)",    "m",                  (-1.5, 1.5),  :balance),
    ("heat_flux",      (f, k) -> -read2d(f, "heat_flux", k),                    "Net surface heat flux into the ocean",        "W m⁻²",              (-250, 250),  :balance),
    ("fw_flux",        (f, k) -> -1e5 .* read2d(f, "fw_flux", k),               "Net surface freshwater flux into the ocean",  "kg m⁻² s⁻¹ × 10⁵",   (-10, 10),    :balance),
    ("mld",            (f, k) -> read2d(f, "mld", k),                           "Mixed layer depth",                           "m",                  (0, 500),     :deep),
    ("w_top",          (f, k) -> 86400 .* read2d(f, "w_0000m", k),              "Vertical velocity at the top level (up > 0)", "m day⁻¹",            (-1, 1),      :balance),
    ("eke",            eke,                                                     "Surface eddy kinetic energy",                 "m² s⁻²",             (1e-3, 0.5),  :inferno),
]
const LOG_FIELDS = ("eke",)
# ANIM_FIELDS="surface_speed,eke" (comma-separated names) animates only those fields; unset animates all
const SELECTED = filter(F -> (sel = get(ENV, "ANIM_FIELDS", ""); isempty(sel) || F[1] in split(sel, ",")), FIELDS)

"(path, key, t) of every output in `paths` holding `var`, in time order; where files overlap (a restart), the later file wins"
function frame_list(paths, var)
    fr = Tuple{String, String, Float64}[]
    for p in paths
        jldopen(p, "r") do f
            (haskey(f, "timeseries/t") && haskey(f, "timeseries/$var")) || return
            for k in frames(f, "t"); push!(fr, (p, k, f["timeseries/t/$k"])); end
        end
    end
    day(x) = round(Int, x[3] / 86400)
    keep = Dict(day(x) => x for x in fr)     # later files overwrite earlier ones
    return sort(collect(values(keep)), by = x -> x[3])
end

"apply `f(file, key, n)` to each frame, opening each file once"
function each_frame(f, fr)
    n = 0
    for p in unique(first.(fr))
        jldopen(p, "r") do fh
            for x in fr; x[1] == p || continue; n += 1; f(fh, x[2], n); end
        end
    end
end

yeartag(fr, window) = (y(x) = floor(Int, (x[3] - window / 2) / YR); @sprintf("years%02d-%02d", y(fr[1]), y(fr[end])))

"all global fields in one pass over the files: one figure and video stream per field"
function animate_fields(paths; fps = 24)
    fr = frame_list(paths, "T_0000m"); isempty(fr) && return
    W = 5 * 86400; tag = yeartag(fr, W)
    if any(F -> F[1] == "eke", SELECTED)   # first pass: time-mean surface velocity over the window
        UBAR[] = zeros(NX, NY); VBAR[] = zeros(NX, NY)
        each_frame(fr) do fh, k, n
            uc, vc = centred_velocity(fh, k); UBAR[] .+= uc ./ length(fr); VBAR[] .+= vc ./ length(fr)
        end
    end
    figs = map(SELECTED) do (name, reader, title, unit, cr, cmap)
        obs = Observable(fill(NaN, length(GLON), length(GLAT))); lab = Observable("")
        fig = Figure(size = (1600, 860), fontsize = 18)
        Label(fig[0, 1:2], title; fontsize = 24, font = :bold, tellwidth = false)
        ax = Axis(fig[1, 1]; aspect = DataAspect(), title = lab, titlefont = :regular,
                  xticks = -180:60:180, yticks = -60:30:90, xtickformat = "{:d}°", ytickformat = "{:d}°")
        cm = Makie.to_colormap(cmap)
        hm = heatmap!(ax, GLON, GLAT, obs; colorrange = cr, colormap = cmap, colorscale = name in LOG_FIELDS ? log10 : identity,
                      nan_color = :gray75, lowclip = cm[1], highclip = cm[end])
        xlims!(ax, -180, 180); ylims!(ax, -80, 88)
        Colorbar(fig[1, 2], hm; label = unit, height = Relative(0.8))
        Label(fig[2, 1:2], "NumericalEarth ⅙° RYF control, 5-day means, binned to 0.25°" *
                           (name == "eke" ? "; EKE about the time mean of the years shown" : ""); fontsize = 13, color = :gray35, tellwidth = false)
        (; name, reader, obs, lab, vs = VideoStream(fig; framerate = fps, format = "mp4"))
    end
    each_frame(fr) do fh, k, n
        label = datelabel(fh["timeseries/t/$k"], W)
        for F in figs
            F.obs[] = regrid(F.reader(fh, k)); F.lab[] = label
            recordframe!(F.vs)
        end
        n % 73 == 0 && println("  frame $n / $(length(fr))  ($label)")
    end
    for F in figs
        out = joinpath(OUTDIR, "$(F.name)_$tag.mp4"); save(out, F.vs); println("saved ", out, " (", length(fr), " frames)")
    end
end

function animate_ice(paths; fps = 6)
    fr = frame_list(paths, "ice_concentration"); isempty(fr) && return
    W = YR / 12; tag = yeartag(fr, W)
    on_ = Observable(fill(NaN, NP, NP)); os_ = Observable(fill(NaN, NP, NP)); lab = Observable("")
    fig = Figure(size = (1500, 820), fontsize = 18)
    Label(fig[0, 1:3], lab; fontsize = 24, font = :bold, tellwidth = false)
    for (c, o, t) in ((1, on_, "Arctic"), (2, os_, "Antarctic"))
        ax = Axis(fig[1, c]; aspect = DataAspect(), title = t)
        hidedecorations!(ax); hidespines!(ax)
        heatmap!(ax, PX, PX, o; colorrange = (0, 1), colormap = Reverse(:ice), nan_color = :gray75)
        for r in (10, 20, 30, 40)   # latitude circles every 10°
            lines!(ax, r .* cosd.(0:2:360), r .* sind.(0:2:360); color = (:black, 0.25), linewidth = 0.8)
        end
    end
    Colorbar(fig[1, 3], colorrange = (0, 1), colormap = Reverse(:ice), label = "sea-ice concentration", height = Relative(0.7))
    Label(fig[2, 1:3], "NumericalEarth ⅙° RYF control, monthly means; polar azimuthal projection, circles every 10° of latitude to 50°"; fontsize = 13, color = :gray35, tellwidth = false)
    vs = VideoStream(fig; framerate = fps, format = "mp4")
    each_frame(fr) do fh, k, n
        A = read2d(fh, "ice_concentration", k)
        on_[] = reshape(apply(BN, A), NP, NP); os_[] = reshape(apply(BS, A), NP, NP)
        lab[] = "Sea-ice concentration, " * datelabel(fh["timeseries/t/$k"], W)
        recordframe!(vs)
    end
    out = joinpath(OUTDIR, "sea_ice_concentration_$tag.mp4"); save(out, vs); println("saved ", out, " (", length(fr), " frames)")
end

# files in time order (later files win where a restart overlaps); fields and sea-ice files told apart by name
animate_ice(filter(p -> occursin("sea_ice", basename(p)), ARGS))
animate_fields(filter(p -> occursin("fields", basename(p)), ARGS))
