using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2
using Glob

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

resolution = "sxtdeg"

function copy_files_to_tempdir(files::Vector{String}; prefix::String)
    copy_dir = mktempdir(; prefix)
    copied = String[]

    try
        for file in files
            dest = joinpath(copy_dir, basename(file))
            cp(file, dest; force = true)
            push!(copied, dest)
        end
    catch
        rm(copy_dir; recursive = true, force = true)
        rethrow()
    end

    @info "Copied analysis inputs." source_files = length(files) copy_dir
    return copied, copy_dir
end

function cleanup_copied_outputs!(copy_dir::Union{Nothing, String})
    if copy_dir !== nothing && isdir(copy_dir)
        rm(copy_dir; recursive = true, force = true)
        @info "Deleted copied analysis inputs." copy_dir
    end
    return nothing
end

run_number(path::AbstractString) = begin
    m = match(r"run(\d+)", basename(path))
    m === nothing ? -1 : parse(Int, m.captures[1])
end

# Example: get all matching files in a folder
files = glob("global_tot*$(resolution)_RYF_run*.jld2", output_path)

# --- Extract iteration numbers ---
iterations = [run_number(f) for f in files if run_number(f) >= 0]
unique_iterations = sort(unique(iterations))

tot_files = String[]

for iteration in unique_iterations
    it_id = lpad(iteration, 4, '0')
    pattern = "global_tot_integrals_$(resolution)_RYF_run$(it_id).jld2"
    matching_files = glob(pattern, output_path)
    if !isempty(matching_files)
        push!(tot_files, matching_files[1])
    else
        @warn "No files found for iteration: $iteration"
    end
end

sort!(tot_files; by = run_number)

copied_tot_dir = nothing
tot_files, copied_tot_dir = copy_files_to_tempdir(tot_files; prefix = "analysis2d_vertical_")
atexit(() -> cleanup_copied_outputs!(copied_tot_dir))

 vars_vertint = ["T_vertintegral",
 "S_vertintegral",
 "e_vertintegral",
 "u_vertintegral",
 "v_vertintegral",
 "w_vertintegral"]
 vols_vertint = ["vert_volume_c",
 "vert_volume_x",
 "vert_volume_y"]
 time = ["t"]

vars = vcat(vars_vertint, vols_vertint)
Lz = jldopen(tot_files[1], "r") do data0
    data0["grid/underlying_grid/z/cᵃᵃᶜ"][7:end-7]
end

using JLD2

function create_dict(vars, path)
    dicts = Dict{String, Any}()
    run = run_number(path)
    data = jldopen(path, "r")
    try
        for var in vars
            try
                grp = data["timeseries/" * var]
                num_keys = filter(k -> tryparse(Int, k) !== nothing, collect(keys(grp)))
                iterations = sort(parse.(Int, num_keys))

                # read actual times for these iterations
                tgrp = data["timeseries/t/"]
                times = Float64[tgrp[string(it)] for it in iterations]

                # read values (vertical profiles), stack to (nt × nz)
                values_vert = Vector{Vector{Float32}}(undef, length(iterations))
                for (i, it) in enumerate(iterations)
                    v = grp[string(it)][1, 1, :][7:end-7]
                    values_vert[i] = vec(v)
                end
                values_matrix = permutedims(hcat(values_vert...))   # (nt × nz)

                dicts[var] = (
                    run = run,
                    iterations = iterations,
                    times      = times,
                    values     = values_matrix
                )
            catch e
                if e isa KeyError
                    @warn "Skipping variable $var: Key not found in file."
                else
                    rethrow(e)
                end
            end
        end
    finally
        close(data)
    end
    return dicts
end

@info "I am loading the surface" 
slice_times = Any[]
for file in tot_files
    push!(slice_times, create_dict(vars, file))
end

concatted_timeseries = Dict{String, Any}()

for var in vars
    by_time = Dict{Float64, NamedTuple{(:run, :iter, :value), Tuple{Int, Int, Vector{Float32}}}}()
    replaced_duplicates = 0

    for slice in slice_times
        run = slice[var].run
        for i in eachindex(slice[var].times)
            t = slice[var].times[i]
            iter = slice[var].iterations[i]
            value = collect(@view slice[var].values[i, :])

            existing = get(by_time, t, nothing)
            if isnothing(existing) || run > existing.run || (run == existing.run && iter >= existing.iter)
                replaced_duplicates += !isnothing(existing) && run > existing.run ? 1 : 0
                by_time[t] = (run = run, iter = iter, value = value)
            end
        end
    end

    uniq_times = sort(collect(keys(by_time)))
    isempty(uniq_times) && error("No timesteps found after deduplication for variable $var.")

    nlevels = length(by_time[uniq_times[1]].value)
    uniq_values = Matrix{Float32}(undef, length(uniq_times), nlevels)
    uniq_iters = Vector{Int}(undef, length(uniq_times))
    for (i, t) in enumerate(uniq_times)
        uniq_iters[i] = by_time[t].iter
        uniq_values[i, :] = by_time[t].value
    end

    concatted_timeseries[var] = (
        iterations = uniq_iters,
        times      = uniq_times,
        values     = uniq_values
    )
    @info "Merged variable with run-priority deduplication." variable = var unique_steps = length(uniq_times) replaced_duplicates
end

totint = Dict(var => concatted_timeseries[var].values for var in vars)

# Use times from one representative var (they should all match after dedupe)
times = concatted_timeseries[vars[1]].times

for var in vars
    @assert concatted_timeseries[var].iterations == concatted_timeseries[vars[1]].iterations
    @assert concatted_timeseries[var].times      == concatted_timeseries[vars[1]].times
end

time_day = times ./ (3600 * 24)
time_in_years = time_day ./ 365

fig = Figure(size = (1100, 700))
ax1 = Axis(fig[1, 1], title = "OHC", xlabel = "Time (years)", ylabel = "Depth (m)")
ax2 = Axis(fig[2, 1], title = "Mean Temperature", xlabel = "Time (years)", ylabel = "Depth (m)")
ax3 = Axis(fig[1, 3], title = "OSC", xlabel = "Time (years)", ylabel = "Depth (m)")
ax4 = Axis(fig[2, 3], title = "Mean Salinity", xlabel = "Time (years)", ylabel = "Depth (m)")
ax5 = Axis(fig[3, 1:3], title = "KE analog", xlabel = "Time (years)", ylabel = "Depth (m)")

# --- pick "first year" indices robustly ---
t0 = minimum(time_in_years)
base_idx = findall(t -> (t ≥ t0) && (t < t0 + 1.0), time_in_years)

@assert !isempty(base_idx) "No samples found in the first year window."

function mean_ignore_nan(A; dims=1)
    good = .!isnan.(A)
    s = sum(replace(A, NaN => zero(eltype(A))); dims=dims)
    n = sum(good; dims=dims)
    return s ./ n
end

# --- helper: anomaly from first-year mean, computed per depth (dims=1 -> time) ---
anom_firstyear(A) = A .- mean_ignore_nan(A[base_idx, :], dims=1)

# --- build the 5 plotted fields (nt × nz) ---
OHC  = 1035 * 1000 .* totint[vars[1]]
Tbar = totint[vars[1]] ./ totint[vars[7]]
OSC  = (1.3358605008598876e18 .- totint[vars[2]]) ./ 35
Sbar = totint[vars[2]] ./ totint[vars[7]]
KE   = 0.5 .* (totint[vars[4]].^2 .+ totint[vars[5]].^2)

# --- anomalies relative to first-year mean ---
OHCa  = anom_firstyear(OHC)
Tbara = anom_firstyear(Tbar)
OSCa  = anom_firstyear(OSC)
Sbara = anom_firstyear(Sbar)
KEa   = anom_firstyear(KE)

# --- plot anomalies ---
hm1 = heatmap!(ax1, time_in_years, Lz, OHCa,  label = "OHC anomaly (vs first-year mean)", colormap=:bwr)
hm2 = heatmap!(ax2, time_in_years, Lz, Tbara, label = "Mean Temperature anomaly (vs first-year mean)", colormap=:bwr)
hm3 = heatmap!(ax3, time_in_years, Lz, OSCa,  label = "OSC anomaly (vs first-year mean)", colormap=:bwr)
hm4 = heatmap!(ax4, time_in_years, Lz, Sbara, label = "Mean Salinity anomaly (vs first-year mean)", colormap=:bwr)
hm5 = heatmap!(ax5, time_in_years, Lz, KEa,   label = "Total KE anomaly (vs first-year mean)", colormap=:bwr)

Colorbar(fig[1, 2], hm1, label = "OHC anomaly")
Colorbar(fig[2, 2], hm2, label = "T anomaly")
Colorbar(fig[1, 4], hm3, label = "OSC anomaly")
Colorbar(fig[2, 4], hm4, label = "S anomaly")
Colorbar(fig[3, 4], hm5, label = "KE anomaly")

function final_times_from_files(tot_files)
    final_iters = Int[]
    final_times = Float64[]

    for file in tot_files
        data = jldopen(file, "r")
        try
            tgrp = data["timeseries/t/"]
            # iteration keys are stored as strings; keep only those that parse to Int
            num_keys = filter(k -> tryparse(Int, k) !== nothing, collect(keys(tgrp)))
            iters = parse.(Int, num_keys)
            it_final = maximum(iters)

            t_final = Float64(tgrp[string(it_final)])  # seconds (usually)
            push!(final_iters, it_final)
            push!(final_times, t_final)
        finally
            close(data)
        end
    end

    return final_iters, final_times
end

final_iters, final_times = final_times_from_files(tot_files)

final_time_in_years = (final_times ./ (3600 * 24)) ./ 365

for ax in (ax1, ax2, ax3, ax4, ax5)
    vlines!(ax, final_time_in_years; linestyle = :dash, color=:black)
end

ylims!(ax1, -1000, 0)
ylims!(ax2, -1000, 0)
ylims!(ax3, -1000, 0)
ylims!(ax4, -1000, 0)
ylims!(ax5, -1000, 0)

save(figdir * "integrated_props_z_$(resolution).png", fig, px_per_unit=3)
cleanup_copied_outputs!(copied_tot_dir)
copied_tot_dir = nothing
