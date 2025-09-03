using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2


output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

tot_files = ["global_tot_integrals_onedeg_RYF_iteration0.jld2", "global_tot_integrals_onedeg_RYF_iteration360.jld2"]
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

concatted_timeseries = Dict()
for var in vars
    concatted_timeseries[var] = cat(getindex.(slice_times, var)..., dims=1)
end
# Assume slice["T"] is your FieldTimeSeries 

T_avg = concatted_timeseries["T_avg"]
S_avg = concatted_timeseries["S_avg"]
u_avg = concatted_timeseries["u_avg"]
v_avg = concatted_timeseries["v_avg"]
w_avg = concatted_timeseries["w_avg"]

T_int = concatted_timeseries["T_totintegral"]
S_int = concatted_timeseries["S_totintegral"]
u_int = concatted_timeseries["u_totintegral"]
v_int = concatted_timeseries["v_totintegral"]
w_int = concatted_timeseries["w_totintegral"]

time_day = 1:size(T_avg)[1]
time_year = time_day

fig = Figure(size = (1200, 800))
# 1. Temperature
ax1 = Axis(fig[1, 1:3], title = "Temperature", xlabel = "Year", ylabel = "Average Temperature (°C)")
lines!(ax1, time_year, T_avg)
xlims!(ax1, 0, maximum(time_year))
ylims!(ax1, minimum(T_avg), maximum(T_avg))

# 2. Salinity
ax2 = Axis(fig[1, 4:6], title = "Salinity", xlabel = "Year", ylabel = "Average Salinity (psu)")
lines!(ax2, time_year, S_avg)
xlims!(ax2, 0, maximum(time_year))
ylims!(ax2, minimum(S_avg), maximum(S_avg))

# 3. U velocity
ax3 = Axis(fig[2, 1:2], title = "U velocity", xlabel = "Year", ylabel = "Average U (m/s)")
lines!(ax3, time_year, u_avg)
xlims!(ax3, 0, maximum(time_year))
ylims!(ax3, minimum(u_avg), maximum(u_avg))

# 4. V velocity
ax4 = Axis(fig[2, 3:4], title = "V velocity", xlabel = "Year", ylabel = "Average V (m/s)")
lines!(ax4, time_year, v_avg)
xlims!(ax4, 0, maximum(time_year))
ylims!(ax4, minimum(v_avg), maximum(v_avg))

# 5. W velocity
ax5 = Axis(fig[2, 5:6], title = "W velocity", xlabel = "Year", ylabel = "Average W (m/s)")
lines!(ax5, time_year, w_avg)
xlims!(ax5, 0, maximum(time_year))
ylims!(ax5, minimum(w_avg), maximum(w_avg))

save(figdir * "average_global_vars_new.png", fig, px_per_unit=3)

fig = Figure(size = (1200, 800))
# 1. Temperature
ax1 = Axis(fig[1, 1:3], title = "Temperature", xlabel = "Year", ylabel = "Integrated Temperature (°C)")
lines!(ax1, time_year, T_int)
xlims!(ax1, 0, maximum(time_year))
ylims!(ax1, minimum(T_int), maximum(T_int))

# 2. Salinity
ax2 = Axis(fig[1, 4:6], title = "Salinity", xlabel = "Year", ylabel = "Integrated Salinity (psu)")
lines!(ax2, time_year, S_int)
xlims!(ax2, 0, maximum(time_year))
ylims!(ax2, minimum(S_int), maximum(S_int))

# 3. U velocity
ax3 = Axis(fig[2, 1:2], title = "U velocity", xlabel = "Year", ylabel = "Integrated U (m/s)")
lines!(ax3, time_year, u_int)
xlims!(ax3, 0, maximum(time_year))
ylims!(ax3, minimum(u_int), maximum(u_int))

# 4. V velocity
ax4 = Axis(fig[2, 3:4], title = "V velocity", xlabel = "Year", ylabel = "Integrated V (m/s)")
lines!(ax4, time_year, v_int)
xlims!(ax4, 0, maximum(time_year))
ylims!(ax4, minimum(v_int), maximum(v_int))

# 5. W velocity
ax5 = Axis(fig[2, 5:6], title = "W velocity", xlabel = "Year", ylabel = "Integrated W (m/s)")
lines!(ax5, time_year, w_int)
xlims!(ax5, 0, maximum(time_year))
ylims!(ax5, minimum(w_int), maximum(w_int))

save(figdir * "int_global_vars_new.png", fig, px_per_unit=3)
