using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2
using Glob

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved_fields/onedeg/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

resolution = "onedeg"

# Example: get all matching files in a folder
files_totint = glob("global_tot*$(resolution)_RYF_run*.jld2", output_path)
files_surface = glob("*forcing_field*$(resolution)_RYF_run*.jld2", output_path)
files = vcat([files_totint,files_surface])
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

vars_totint = ["T_totintegral",
 "S_totintegral",
 "e_totintegral",
 "u_totintegral",
 "v_totintegral",
 "w_totintegral"]
#  vars_vertint = ["T_vertintegral",
#  "S_vertintegral",
#  "e_vertintegral",
#  "u_vertintegral",
#  "v_vertintegral",
#  "w_vertintegral"]
 vols_totint = ["total_volume_c",
 "total_volume_x",
 "total_volume_y"]
#  vols_vertint = ["vert_volume_c",
#  "vert_volume_x",
#  "vert_volume_y"]
 time = ["t"]

vars = vcat(vars_totint, vols_totint, time)

function create_dict(vars, path)
    dicts = Dict()
    data = jldopen(path)
    for var in vars
        try
            grp = data["timeseries/" * var]
            num_keys = filter(k -> tryparse(Int, k) !== nothing, keys(grp))
            timesteps = sort(parse.(Int, num_keys))
            values = [grp[string(t)] for t in timesteps]
            dicts[var] = (
                timesteps = timesteps,
                values = only.(values)
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

totint = Dict()

for var in vars
    totint[var] = concatted_timeseries[var].values
end
time_day = totint["t"]/(3600*24)
time_in_years = time_day/365

fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
ax2 = Axis(fig[2, 1], title = "Mean Temperature", xlabel = "Time (years)", ylabel = "Temperature (°C)")
ax3 = Axis(fig[1, 2], title = "OSC", xlabel = "Time (years)", ylabel = "OFWC (kg)")
ax4 = Axis(fig[2, 2], title = "Mean Salinity", xlabel = "Time (years)", ylabel = "Salinity (psu)")
ax5 = Axis(fig[3, :], title = "Total Volume", xlabel = "Time (years)", ylabel = "Volume (m³)")

lines!(ax1, time_in_years, 1035*1000*totint[vars[1]], label = "OHC")
lines!(ax2, time_in_years, totint[vars[1]]./totint[vars[7]], label = "Mean Temperature")
lines!(ax3, time_in_years, (1.3358605008598876e18.-totint[vars[2]])./(35), label = "OSC")
lines!(ax4, time_in_years, totint[vars[2]]./totint[vars[7]], label = "Mean Salinity")
lines!(ax5, time_in_years, 0.5*(totint[vars[4]].^2+totint[vars[5]].^2+totint[vars[6]].^2), label = "Total KE")

save(figdir * "integrated_props_$(resolution).png", fig, px_per_unit=3)

### Now we work on the heat budget!

