# Reduce closed output files to the post-processed set kept on the project disk.
# Usage: julia --project=<experiments/submission_scripts> postprocess_chunk.jl <postprocessed_run_dir> <file.jld2>...
#
# For every file, writes into <postprocessed_run_dir>:
#   timeseries/<stem>.jld2   scalar outputs and area means of every 2D output, one value per record
#   zonal/<stem>.jld2        area-weighted zonal means of every 2D output on 0.25° latitude bins, per record
#   annual/year_NNNN.jld2    running annual-mean maps of every 2D output (merged across files, records counted once)
#   seasonal/year_NNNN.jld2  running quarterly (JFM, AMJ, JAS, OND) mean maps of SEASONAL_VARS, keys <var>_Q1..Q4
# PP_ONLY_SEASONAL=1 only (re)builds the seasonal maps (backfill from files fetched back from Mediaflux).
# Prints "POSTPROCESSED <file>" for each file that completed, which storage_watcher.sh requires before deleting it.

using JLD2
using Oceananigans   # so the serialized grid is rebuilt with its real array types
using Logging
using Printf

const PP_DIR = ARGS[1]
const FILES = ARGS[2:end]
const YEAR = 365 * 86400.0
const LAT_EDGES = -80:0.25:90
const LAT = collect(LAT_EDGES[1:end-1] .+ 0.125)
const LEGACY_LEVEL_DEPTHS = Dict(75 => 0, 67 => 30, 56 => 100, 48 => 200, 36 => 500, 26 => 1000)
const GRID_COORDS = joinpath(dirname(rstrip(PP_DIR, '/')), "grid_coords.jld2")
const SEASONAL_VARS = Set(["mld", "ssh", "T_0000m", "S_0000m", "heat_flux", "fw_flux", "ice_concentration", "u_0000m", "v_0000m"])
const ONLY_SEASONAL = get(ENV, "PP_ONLY_SEASONAL", "0") == "1"
const MONTH_EDGES = cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

save_dict(path, d) = jldopen(fh -> foreach(k -> fh[k] = d[k], keys(d)), path, "w")
load_dict(path) = jldopen(fh -> Dict{String, Any}(k => fh[k] for k in keys(fh)), path)

record_keys(fh) = sort(parse.(Int, filter(k -> k != "serialized", collect(keys(fh["timeseries/t"])))))

# Horizontal coordinates, cell areas and sea floor, taken once from an ocean (immersed) grid
function grid_coords(file)
    isfile(GRID_COORDS) && return load_dict(GRID_COORDS)
    coords = with_logger(NullLogger()) do
        jldopen(file) do fh
            grid = fh["serialized/grid"]
            hasproperty(grid, :immersed_boundary) || return nothing
            ug = grid.underlying_grid
            inner(A) = (P = parent(A); ndims(P) == 3 ? P[ug.Hx+1:ug.Hx+ug.Nx, ug.Hy+1:ug.Hy+ug.Ny, 1] : P[ug.Hx+1:ug.Hx+ug.Nx, ug.Hy+1:ug.Hy+ug.Ny])
            Dict("lon" => Float64.(inner(ug.λᶜᶜᵃ)), "lat" => Float64.(inner(ug.φᶜᶜᵃ)),
                 "area" => Float64.(inner(ug.Azᶜᶜᵃ)), "bottom" => Float64.(inner(grid.immersed_boundary.bottom_height)))
        end
    end
    isnothing(coords) && return nothing
    mkpath(dirname(GRID_COORDS))
    save_dict(GRID_COORDS, coords)
    return coords
end

# Slices are named <var>_<DDDD>m; everything else is a surface or column quantity
output_depth(name) = (m = match(r"_(\d{4})m$", name); isnothing(m) ? 0.0 : parse(Float64, m.captures[1]))

function process(file, coords)
    stem = replace(basename(file), r"\.jld2$" => "")
    area, bottom, φ = coords["area"], coords["bottom"], coords["lat"]
    Nx, Ny = size(area)
    lat_bin = clamp.(searchsortedlast.(Ref(LAT_EDGES), φ), 1, length(LAT))

    fh = jldopen(file)
    # A file opened just before a split (e.g. at a restart) can hold no records; nothing to reduce
    haskey(fh, "timeseries/t") || (close(fh); return nothing)
    iters = record_keys(fh)
    times = [Float64(fh["timeseries/t/$i"]) for i in iters]
    names = filter(n -> n != "t", collect(keys(fh["timeseries"])))
    spacing = length(times) > 2 ? sort(diff(times))[cld(length(times) - 1, 2)] : Inf
    partial_first = length(times) > 2 && times[2] - times[1] < 0.9 * spacing

    series = Dict{String, Any}("t" => times)
    zonal = Dict{String, Any}("t" => times, "lat" => LAT)
    annual = Dict{Int, Dict{String, Any}}()
    seasonal = Dict{Int, Dict{String, Any}}()
    stored_times = Dict{Tuple{String, Int}, Dict{String, Any}}()
    function counted_times(kind, year, name)
        path = joinpath(PP_DIR, kind, @sprintf("year_%04d.jld2", year))
        stored = get!(stored_times, (kind, year)) do
            isfile(path) ? jldopen(fh -> Dict(k => fh[k] for k in keys(fh) if endswith(k, "_times")), path) : Dict{String, Any}()
        end
        return get(stored, name * "_times", Float64[])
    end
    function accumulate!(acc, key, A, τ)
        if haskey(acc, key)
            acc[key] .+= A; push!(acc[key * "_times"], τ)
        else
            acc[key] = Float64.(A); acc[key * "_times"] = [τ]
        end
    end

    # Files written before 2026-11-12 hold one level each (global_<k>_fields_*), with variables named T, S, u, v, w
    legacy = match(r"^global_(\d+)_fields", stem)
    outname(name) = isnothing(legacy) ? name : name * "_" * lpad(string(LEGACY_LEVEL_DEPTHS[parse(Int, legacy.captures[1])]), 4, '0') * "m"

    for name in names
        key = outname(name)
        ONLY_SEASONAL && !(key in SEASONAL_VARS) && continue
        first_record = fh["timeseries/$name/$(iters[1])"]
        if length(first_record) == 1
            series[key] = [Float64(only(fh["timeseries/$name/$i"])) for i in iters]
            continue
        end

        wet = bottom .< -max(output_depth(key), 0.5)
        Z = fill(NaN32, length(LAT), length(iters))
        means = zeros(length(iters))
        for (n, i) in enumerate(iters)
            A = fh["timeseries/$name/$i"][:, 1:Ny, 1]     # v has an extra northern face row
            num = zeros(length(LAT)); den = zeros(length(LAT))
            @inbounds for I in eachindex(A)
                (wet[I] && isfinite(A[I])) || continue
                b = lat_bin[I]; num[b] += A[I] * area[I]; den[b] += area[I]
            end
            Z[:, n] .= [d > 0 ? n′ / d : NaN for (n′, d) in zip(num, den)]
            means[n] = sum(num) / sum(den)

            year = floor(Int, (times[n] - 1) / YEAR)       # records are stamped at the end of their window
            times[n] > 0 || continue                          # the t = 0 record is the initial snapshot
            n == 1 && partial_first && continue               # a short window written just after a restart
            if key in SEASONAL_VARS
                # quarter of the window centre (records are stamped at the window end)
                day = mod(times[n] - min(spacing, 31 * 86400.0) / 2, YEAR) / 86400
                q = cld(clamp(searchsortedlast(MONTH_EDGES, day), 1, 12), 3)
                skey = key * "_Q$q"
                times[n] in counted_times("seasonal", year, skey) || accumulate!(get!(seasonal, year, Dict{String, Any}()), skey, A, times[n])
            end
            ONLY_SEASONAL && continue
            times[n] in counted_times("annual", year, key) && continue   # already in the annual mean (rerun or split)
            accumulate!(get!(annual, year, Dict{String, Any}()), key, A, times[n])
        end
        zonal[key] = Z
        series[key * "_areamean"] = means
    end
    close(fh)

    mkpath(joinpath(PP_DIR, "timeseries")); mkpath(joinpath(PP_DIR, "zonal")); mkpath(joinpath(PP_DIR, "annual")); mkpath(joinpath(PP_DIR, "seasonal"))
    if !ONLY_SEASONAL
        save_dict(joinpath(PP_DIR, "timeseries", stem * ".jld2"), series)
        length(zonal) > 2 && save_dict(joinpath(PP_DIR, "zonal", stem * ".jld2"), zonal)
    end

    # Merge the new records into the annual and seasonal means
    for (kind, accs) in (("annual", annual), ("seasonal", seasonal)), (year, acc) in accs
        path = joinpath(PP_DIR, kind, @sprintf("year_%04d.jld2", year))
        stored = isfile(path) ? load_dict(path) : Dict{String, Any}()
        for name in filter(k -> !endswith(k, "_times"), collect(keys(acc)))
            old_times = get(stored, name * "_times", Float64[])
            n_old = length(old_times); n_new = length(acc[name * "_times"])
            old_mean = n_old > 0 ? Float64.(stored[name]) : zeros(size(acc[name]))
            stored[name] = Float32.((old_mean .* n_old .+ acc[name]) ./ (n_old + n_new))
            stored[name * "_times"] = vcat(old_times, acc[name * "_times"])
        end
        tmp = path * ".tmp"
        save_dict(tmp, stored)
        mv(tmp, path; force=true)
    end
    return nothing
end

# Ocean files first, so the grid coordinates come from an immersed ocean grid
ordered = sort(FILES; by=f -> occursin("sea_ice", basename(f)))
coords = nothing
for file in ordered
    global coords
    try
        coords = something(coords, grid_coords(file), Some(nothing))
        isnothing(coords) && error("no ocean grid coordinates yet ($GRID_COORDS missing)")
        process(file, coords)
        println("POSTPROCESSED ", file)
    catch err
        @error "Post-processing failed" file exception=(err, catch_backtrace())
    end
end
