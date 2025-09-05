using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2


output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

other_file = "global_tot_integrals_onedeg_RYF_iteration0_w_seaice.jld2"

tot_files = ["global_tot_integrals_onedeg_RYF_iteration0_CP_w_seaice.jld2",
            "global_tot_integrals_onedeg_RYF_iteration864_CP_w_seaice.jld2"]

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
            @info var
            grp = data["timeseries/" * var]
            num_keys = filter(k -> tryparse(Int, k) !== nothing, keys(grp))
            timesteps = sort(parse.(Int, num_keys))
            values = [grp[string(t)] for t in timesteps]
            @time dicts[var] = [only(v) for v in values]
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
    slice = create_dict(vars, output_path * file)
    push!(slice_times, slice)
end

@show size(slice_times[1]["T_avg"]), size(slice_times[2]["T_avg"])

concatted_timeseries = Dict()
for var in vars
    # concatted_timeseries[var] = cat(getindex.(slice_times, var)..., dims=1)
    concatted_timeseries[var] = vcat(slice_times[1][var][1:end-1], slice_times[2][var])
end

slice = create_dict(vars, output_path * other_file)
# Assume slice["T"] is your FieldTimeSeries 

T_avg = filter(!isnan,concatted_timeseries["T_avg"])
S_avg = filter(!isnan,concatted_timeseries["S_avg"])
u_avg = filter(!isnan,concatted_timeseries["u_avg"])
v_avg = filter(!isnan,concatted_timeseries["v_avg"])
w_avg = filter(!isnan,concatted_timeseries["w_avg"])

T_int = filter(!isnan,concatted_timeseries["T_totintegral"])
S_int = filter(!isnan,concatted_timeseries["S_totintegral"])
u_int = filter(!isnan,concatted_timeseries["u_totintegral"])
v_int = filter(!isnan,concatted_timeseries["v_totintegral"])
w_int = filter(!isnan,concatted_timeseries["w_totintegral"])

time_day = 1:size(filter(!isnan,T_avg))[1]
time_year = time_day

T_avg2 = filter(!isnan,slice["T_avg"])
S_avg2 = filter(!isnan,slice["S_avg"])
u_avg2 = filter(!isnan,slice["u_avg"])
v_avg2 = filter(!isnan,slice["v_avg"])
w_avg2 = filter(!isnan,slice["w_avg"])

T_int2 = filter(!isnan,slice["T_totintegral"])
S_int2 = filter(!isnan,slice["S_totintegral"])
u_int2 = filter(!isnan,slice["u_totintegral"])
v_int2 = filter(!isnan,slice["v_totintegral"])
w_int2 = filter(!isnan,slice["w_totintegral"])

time_day2 = 1:size(filter(!isnan,T_avg2))[1]
time_year2 = time_day2

fig = Figure(size = (1200, 800))
# 1. Temperature
ax1 = Axis(fig[1, 1:3], title = "Temperature", xlabel = "Day", ylabel = "Average Temperature (°C)")
lines!(ax1, time_year, filter(!isnan,T_avg), label = "Checkpoint")
lines!(ax1, time_year2, filter(!isnan,T_avg2), label = "No Checkpoint")
xlims!(ax1, 0, maximum(time_year))
ylims!(ax1, minimum(T_avg), maximum(T_avg))

# 2. Salinity
ax2 = Axis(fig[1, 4:6], title = "Salinity", xlabel = "Day", ylabel = "Average Salinity (psu)")
lines!(ax2, time_year, filter(!isnan,S_avg), label = "Checkpoint")
lines!(ax2, time_year2, filter(!isnan,S_avg2), label = "No Checkpoint")
xlims!(ax2, 0, maximum(time_year))
ylims!(ax2, minimum(S_avg), maximum(S_avg))

# 3. U velocity
ax3 = Axis(fig[2, 1:2], title = "U velocity", xlabel = "Day", ylabel = "Average U (m/s)")
lines!(ax3, time_year, filter(!isnan,u_avg), label = "Checkpoint")
lines!(ax3, time_year2, filter(!isnan,u_avg2), label = "No Checkpoint")
xlims!(ax3, 0, maximum(time_year))
ylims!(ax3, minimum(u_avg2), maximum(u_avg2))

# 4. V velocity
ax4 = Axis(fig[2, 3:4], title = "V velocity", xlabel = "Day", ylabel = "Average V (m/s)")
lines!(ax4, time_year, filter(!isnan,v_avg), label = "Checkpoint")
lines!(ax4, time_year2, filter(!isnan,v_avg2), label = "No Checkpoint")
xlims!(ax4, 0, maximum(time_year))
ylims!(ax4, minimum(v_avg2), maximum(v_avg2))

# 5. W velocity
ax5 = Axis(fig[2, 5:6], title = "W velocity", xlabel = "Day", ylabel = "Average W (m/s)")
lines!(ax5, time_year, filter(!isnan,w_avg), label = "Checkpoint")
lines!(ax5, time_year2, filter(!isnan,w_avg2), label = "No Checkpoint")
xlims!(ax5, 0, maximum(time_year))
ylims!(ax5, minimum(w_avg2), maximum(w_avg2))

Legend(fig[1, 1], ax1)

save(figdir * "average_global_vars_w_seaice.png", fig, px_per_unit=3)

fig = Figure(size = (1200, 800))
# 1. Temperature
ax1 = Axis(fig[1, 1:3], title = "Temperature", xlabel = "Day", ylabel = "Integrated Temperature (°C)")
lines!(ax1, time_year, filter(!isnan,T_int), label = "Checkpoint")
lines!(ax1, time_year2, filter(!isnan,T_int2), label = "No Checkpoint")
xlims!(ax1, 0, maximum(time_year))
ylims!(ax1, minimum(T_int2), maximum(T_int2))

# 2. Salinity
ax2 = Axis(fig[1, 4:6], title = "Salinity", xlabel = "Day", ylabel = "Integrated Salinity (psu)")
lines!(ax2, time_year, filter(!isnan,S_int), label = "Checkpoint")
lines!(ax2, time_year2, filter(!isnan,S_int2), label = "No Checkpoint")
xlims!(ax2, 0, maximum(time_year))
ylims!(ax2, minimum(S_int2), maximum(S_int2))

# 3. U velocity
ax3 = Axis(fig[2, 1:2], title = "U velocity", xlabel = "Day", ylabel = "Integrated U (m/s)")
lines!(ax3, time_year, filter(!isnan,u_int), label = "Checkpoint")
lines!(ax3, time_year2, filter(!isnan,u_int2), label = "No Checkpoint")
xlims!(ax3, 0, maximum(time_year))
ylims!(ax3, minimum(u_int2), maximum(u_int2))

# 4. V velocity
ax4 = Axis(fig[2, 3:4], title = "V velocity", xlabel = "Day", ylabel = "Integrated V (m/s)")
lines!(ax4, time_year, filter(!isnan,v_int), label = "Checkpoint")
lines!(ax4, time_year2, filter(!isnan,v_int2), label = "No Checkpoint")
xlims!(ax4, 0, maximum(time_year))
ylims!(ax4, minimum(v_int2), maximum(v_int2))

# 5. W velocity
ax5 = Axis(fig[2, 5:6], title = "W velocity", xlabel = "Day", ylabel = "Integrated W (m/s)")
lines!(ax5, time_year, filter(!isnan,w_int), label = "Checkpoint")
lines!(ax5, time_year2, filter(!isnan,w_int2), label = "No Checkpoint")
xlims!(ax5, 0, maximum(time_year))
ylims!(ax5, minimum(w_int2), maximum(w_int2))

Legend(fig[1, 1], ax1)

save(figdir * "int_global_vars_w_seaice.png", fig, px_per_unit=3)
