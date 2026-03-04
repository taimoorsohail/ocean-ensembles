using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2
using Glob

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved_fields/onedeg/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

resolution = "onedeg"

# Example: get all matching files in a folder
files = glob("global_tot*$(resolution)_RYF_run*.jld2", output_path)

# --- Extract iteration numbers ---
iterations = [parse(Int, match(r"run(\d+)", f).captures[1]) 
              for f in files if occursin(r"run\d+", f)]
unique_iterations = sort(unique(iterations))

tot_files = []

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

times = Float64[]
iters = Int[]

for filename in tot_files
    data = jldopen(filename, "r")
    try
        timeiters = sort(parse.(Int, collect(keys(data["timeseries/t/"]))))

        for iter in timeiters
            t = data["timeseries/t/$(iter)"]  # scalar time
            push!(times, Float64(t))
            push!(iters, iter)
        end
    finally
        close(data)
    end
end

# --- keep LAST occurrence of each iter ---
rev_iters = reverse(iters)
idxs_rev  = unique(i -> rev_iters[i], eachindex(rev_iters))   # indices in reversed arrays

# map indices back to original order
idxs = length(iters) .- idxs_rev .+ 1
sort!(idxs)  # chronological by original position

# mask both arrays the same way
iters = iters[idxs]
times = times[idxs]

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
data0 = jldopen(tot_files[1])
Lz = data0["grid/underlying_grid/z/cᵃᵃᶜ"][7:end-7]

using JLD2

function create_dict(vars, path)
    dicts = Dict{String, Any}()
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
                values_matrix = hcat(values_vert...)'   # (nt × nz)

                dicts[var] = (
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
    all_iters  = vcat((slice[var].iterations for slice in slice_times)...)
    all_times  = vcat((slice[var].times      for slice in slice_times)...)
    all_values = vcat((slice[var].values     for slice in slice_times)...)

    @assert length(all_iters) == length(all_times) == size(all_values, 1)

    # reverse so unique keeps the last occurrence in original order
    rev_iters  = reverse(all_iters)
    rev_times  = reverse(all_times)
    rev_values = reverse(all_values; dims=1)

    idxs = unique(i -> rev_iters[i], eachindex(rev_iters))  # indices in reversed arrays

    uniq_iters  = rev_iters[idxs]
    uniq_times  = rev_times[idxs]
    uniq_values = rev_values[idxs, :]

    # flip back to chronological (by original order)
    concatted_timeseries[var] = (
        iterations = reverse(uniq_iters),
        times      = reverse(uniq_times),
        values     = reverse(uniq_values; dims=1)
    )
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
