using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2
using Glob

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

resolution = "onedeg"

# Example: get all matching files in a folder
files = glob("global_tot*$(resolution)_RYF_iteration*.jld2", output_path)

# --- Extract iteration numbers ---
iterations = [parse(Int, match(r"iteration(\d+)", f).captures[1]) 
              for f in files if occursin(r"iteration\d+", f)]
unique_iterations = sort(unique(iterations))

tot_files = []

for iteration in unique_iterations
    pattern = "global_tot_integrals_$(resolution)_RYF_iteration$(iteration).jld2"
    matching_files = glob(pattern, output_path)
    if !isempty(matching_files)
        push!(tot_files, matching_files[1])
    else
        @warn "No files found for iteration: $iteration"
    end

end

vars_int = [ "T_totintegral",
 "S_totintegral",
 "u_totintegral",
 "v_totintegral",
 "w_totintegral"]

 vars_avg = [ "T_avg",
 "S_avg",
 "u_avg",
 "v_avg",
 "w_avg"]

vars = vcat(vars_int, vars_avg)

function create_dict(vars, path)
    dicts = Dict()
    data = jldopen(path)
    for var in vars
        try
            # Surface
            grp = data["timeseries/" * var]
            num_keys = filter(k -> tryparse(Int, k) !== nothing, keys(grp))
            timesteps = sort(parse.(Int, num_keys))
            values = [grp[string(t)] for t in timesteps]
            dicts[var] = (
                timesteps = timesteps,
                values = [only(v) for v in values]
)
        catch e
            if e isa KeyError
                @warn "Skipping variable $var: Key not found in file."
            else
                rethrow(e)
            end
        end
    end
    close(data)
    return dicts
end

@info "I am loading the surface" 
slice_times = []
for file in tot_files
    slice = create_dict(vars, file)
    push!(slice_times, slice)
end

concatted_timeseries = Dict()
for var in vars
    # collect all timesteps + values
    all_times  = vcat([slice[var].timesteps for slice in slice_times]...)
    all_values = vcat([slice[var].values    for slice in slice_times]...)

    # reverse so unique() keeps the *last* occurrence
    rev_times  = reverse(all_times)
    rev_values = reverse(all_values)

    uniq_times = unique(rev_times)
    idxs = unique(z -> rev_times[z], 1:length(rev_times))
    uniq_values = rev_values[idxs]

    # flip back to chronological order
    concatted_timeseries[var] = (
        timesteps = reverse(uniq_times),
        values    = reverse(uniq_values)
    )
end

T_avg = filter(!isnan,concatted_timeseries["T_avg"].values)
S_avg = filter(!isnan,concatted_timeseries["S_avg"].values)
u_avg = filter(!isnan,concatted_timeseries["u_avg"].values)
v_avg = filter(!isnan,concatted_timeseries["v_avg"].values)
w_avg = filter(!isnan,concatted_timeseries["w_avg"].values)

T_int = filter(!isnan,concatted_timeseries["T_totintegral"].values)
S_int = filter(!isnan,concatted_timeseries["S_totintegral"].values)
u_int = filter(!isnan,concatted_timeseries["u_totintegral"].values)
v_int = filter(!isnan,concatted_timeseries["v_totintegral"].values)
w_int = filter(!isnan,concatted_timeseries["w_totintegral"].values)

time_day = 1:size(filter(!isnan,T_avg))[1]
time_year = time_day/365

# T_avg2 = filter(!isnan,slice["T_avg"])
# S_avg2 = filter(!isnan,slice["S_avg"])
# u_avg2 = filter(!isnan,slice["u_avg"])
# v_avg2 = filter(!isnan,slice["v_avg"])
# w_avg2 = filter(!isnan,slice["w_avg"])

# T_int2 = filter(!isnan,slice["T_totintegral"])
# S_int2 = filter(!isnan,slice["S_totintegral"])
# u_int2 = filter(!isnan,slice["u_totintegral"])
# v_int2 = filter(!isnan,slice["v_totintegral"])
# w_int2 = filter(!isnan,slice["w_totintegral"])

# time_day2 = 1:size(filter(!isnan,T_avg2))[1]
# time_year2 = time_day2

fig = Figure(size = (1200, 800))
# 1. Temperature
ax1 = Axis(fig[1, 1:3], title = "Temperature", xlabel = "Year", ylabel = "Average Temperature (°C)")
lines!(ax1, time_year, filter(!isnan,T_avg), label = "Checkpoint")
# lines!(ax1, time_year2, filter(!isnan,T_avg2), label = "No Checkpoint")
xlims!(ax1, 0, maximum(time_year))
ylims!(ax1, minimum(T_avg), maximum(T_avg))

# 2. Salinity
ax2 = Axis(fig[1, 4:6], title = "Salinity", xlabel = "Year", ylabel = "Average Salinity (psu)")
lines!(ax2, time_year, filter(!isnan,S_avg), label = "Checkpoint")
# lines!(ax2, time_year2, filter(!isnan,S_avg2), label = "No Checkpoint")
xlims!(ax2, 0, maximum(time_year))
ylims!(ax2, minimum(S_avg), maximum(S_avg))

# 3. U velocity
ax3 = Axis(fig[2, 1:2], title = "U velocity", xlabel = "Year", ylabel = "Average U (m/s)")
lines!(ax3, time_year, filter(!isnan,u_avg), label = "Checkpoint")
# lines!(ax3, time_year2, filter(!isnan,u_avg2), label = "No Checkpoint")
xlims!(ax3, 0, maximum(time_year))
ylims!(ax3, minimum(u_avg), maximum(u_avg))

# 4. V velocity
ax4 = Axis(fig[2, 3:4], title = "V velocity", xlabel = "Year", ylabel = "Average V (m/s)")
lines!(ax4, time_year, filter(!isnan,v_avg), label = "Checkpoint")
# lines!(ax4, time_year2, filter(!isnan,v_avg2), label = "No Checkpoint")
xlims!(ax4, 0, maximum(time_year))
ylims!(ax4, minimum(v_avg), maximum(v_avg))

# 5. W velocity
ax5 = Axis(fig[2, 5:6], title = "W velocity", xlabel = "Year", ylabel = "Average W (m/s)")
lines!(ax5, time_year, filter(!isnan,w_avg), label = "Checkpoint")
# lines!(ax5, time_year2, filter(!isnan,w_avg2), label = "No Checkpoint")
xlims!(ax5, 0, maximum(time_year))
ylims!(ax5, minimum(w_avg), maximum(w_avg))

Legend(fig[1, 1], ax1)

save(figdir * "average_global_vars_$(resolution).png", fig, px_per_unit=3)

fig = Figure(size = (1200, 800))
# 1. Temperature
ax1 = Axis(fig[1, 1:3], title = "Temperature", xlabel = "Year", ylabel = "Integrated Temperature (°C)")
lines!(ax1, time_year, filter(!isnan,T_int), label = "Checkpoint")
# lines!(ax1, time_year2, filter(!isnan,T_int2), label = "No Checkpoint")
xlims!(ax1, 0, maximum(time_year))
ylims!(ax1, minimum(T_int), maximum(T_int))

# 2. Salinity
ax2 = Axis(fig[1, 4:6], title = "Salinity", xlabel = "Year", ylabel = "Integrated Salinity (psu)")
lines!(ax2, time_year, filter(!isnan,S_int), label = "Checkpoint")
# lines!(ax2, time_year2, filter(!isnan,S_int2), label = "No Checkpoint")
xlims!(ax2, 0, maximum(time_year))
ylims!(ax2, minimum(S_int), maximum(S_int))

# 3. U velocity
ax3 = Axis(fig[2, 1:2], title = "U velocity", xlabel = "Year", ylabel = "Integrated U (m/s)")
lines!(ax3, time_year, filter(!isnan,u_int), label = "Checkpoint")
# lines!(ax3, time_year2, filter(!isnan,u_int2), label = "No Checkpoint")
xlims!(ax3, 0, maximum(time_year))
ylims!(ax3, minimum(u_int), maximum(u_int))

# 4. V velocity
ax4 = Axis(fig[2, 3:4], title = "V velocity", xlabel = "Year", ylabel = "Integrated V (m/s)")
lines!(ax4, time_year, filter(!isnan,v_int), label = "Checkpoint")
# lines!(ax4, time_year2, filter(!isnan,v_int2), label = "No Checkpoint")
xlims!(ax4, 0, maximum(time_year))
ylims!(ax4, minimum(v_int), maximum(v_int))

# 5. W velocity
ax5 = Axis(fig[2, 5:6], title = "W velocity", xlabel = "Year", ylabel = "Integrated W (m/s)")
lines!(ax5, time_year, filter(!isnan,w_int), label = "Checkpoint")
# lines!(ax5, time_year2, filter(!isnan,w_int2), label = "No Checkpoint")
xlims!(ax5, 0, maximum(time_year))
ylims!(ax5, minimum(w_int), maximum(w_int))

Legend(fig[1, 1], ax1)

save(figdir * "int_global_vars_$(resolution).png", fig, px_per_unit=3)
