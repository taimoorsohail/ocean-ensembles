using CairoMakie
using Oceananigans  # From local
using Oceananigans.Operators: Az
using OceanEnsembles
using Statistics
using JLD2
using Glob

outputpath = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved_fields/onedeg/")
fig_dir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

resolution = "onedeg"

# Example: get all matching files in a folder
files_totint = glob("global_tot*$(resolution)_RYF_run*.jld2", outputpath)
files_surface = glob("*forcing_field*$(resolution)_RYF_run*.jld2", outputpath)

files = vcat(files_totint, files_surface)
run_id(path) = parse(Int, match(r"run(\d+)", path).captures[1])
sort!(files_surface; by = run_id)
# --- Extract iteration numbers ---
iterations = [parse(Int, match(r"run(\d+)", f).captures[1]) 
              for f in files if occursin(r"run\d+", f)]
unique_iterations = sort(unique(iterations))

tot_files = []

for iteration in unique_iterations
    it_id = lpad(iteration, 4, '0')
    pattern = "global_tot_integrals_$(resolution)_RYF_run$(it_id).jld2"
    matching_files = glob(pattern, outputpath)
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
 vols_totint = ["total_volume_c",
 "total_volume_x",
 "total_volume_y"]
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

function surface_height_timeseries(surface_files, grid)
    all_t = Int[]
    all_mean = Float64[]
    η = Field{Center, Center, Nothing}(grid)
    one = Field{Center, Center, Nothing}(grid)
    set!(one, 1)
    area_total = (Integral(one * Az, dims = (1, 2)) |> Field)[1, 1, 1]

    for file in surface_files
        data = jldopen(file)
        haskey(data, "timeseries/surface_height") || begin
            close(data)
            @warn "Skipping $file: timeseries/surface_height not found."
            continue
        end

        ts_keys = sort(parse.(Int, collect(keys(data["timeseries/t"]))))
        for key in ts_keys
            tval = round(Int, Float64(data["timeseries/t/$key"]))
            raw = data["timeseries/surface_height/$key"]
            η_xy = ndims(raw) == 3 ? raw[:, :, 1] : raw
            set!(η, η_xy)
            η_norm_int = (Integral((η * Az) / area_total, dims = (1, 2)) |> Field)[1, 1, 1]
            push!(all_t, tval)
            push!(all_mean, Float64(η_norm_int))
        end
        close(data)
    end

    order = sortperm(all_t)
    all_t = all_t[order]
    all_mean = all_mean[order]

    rev_t = reverse(all_t)
    rev_mean = reverse(all_mean)
    uniq_idx = unique(i -> rev_t[i], eachindex(rev_t))

    t_unique = reverse(rev_t[uniq_idx])
    mean_unique = reverse(rev_mean[uniq_idx])

    return t_unique ./ (365 * 24 * 60 * 60), mean_unique
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
isempty(files_surface) && error("No forcing surface files found in $outputpath.")
grid_run = lpad(run_id(files_surface[1]), 4, '0')
grid_prefix = outputpath * "global_75_fields_$(resolution)_RYF_run$(grid_run)"
if !isfile(grid_prefix * ".jld2")
    grid_files = glob("global_75_fields_$(resolution)_RYF_run*.jld2", outputpath)
    sort!(grid_files; by = run_id)
    isempty(grid_files) && error("No 75-level run files found in $outputpath.")
    grid_prefix = splitext(last(grid_files))[1]
end
surface_grid = create_grid(grid_prefix; gridtype = "TripolarGrid")
surface_time_in_years, surface_height_mean = surface_height_timeseries(files_surface, surface_grid)

fig = Figure(size = (1000, 700))
ax1 = Axis(fig[1, 1], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
ax2 = Axis(fig[2, 1], title = "Mean Temperature", xlabel = "Time (years)", ylabel = "Temperature (°C)")
ax3 = Axis(fig[1, 2], title = "OSC", xlabel = "Time (years)", ylabel = "OFWC (kg)")
ax4 = Axis(fig[2, 2], title = "Mean Salinity", xlabel = "Time (years)", ylabel = "Salinity (psu)")
ax5 = Axis(fig[3, 1], title = "Total Volume", xlabel = "Time (years)", ylabel = "Volume (m³)")
ax6 = Axis(fig[3, 2], title = "Surface Height (Σ(ηA/ΣA))", xlabel = "Time (years)", ylabel = "Σ(ηA/ΣA)")

lines!(ax1, time_in_years, 1035*1000*totint[vars[1]], label = "OHC")
lines!(ax2, time_in_years, totint[vars[1]]./totint[vars[7]], label = "Mean Temperature")
lines!(ax3, time_in_years, 1035 .* (totint[vars[7]] .- totint[vars[2]] ./ 35), label = "OSC")
lines!(ax4, time_in_years, totint[vars[2]]./totint[vars[7]], label = "Mean Salinity")
lines!(ax5, time_in_years, 0.5*(totint[vars[4]].^2+totint[vars[5]].^2+totint[vars[6]].^2), label = "Total KE analog")
lines!(ax6, surface_time_in_years[2:end], surface_height_mean[2:end], label = "GMSL (m)")

save(fig_dir * "integrated_props_$(resolution).png", fig, px_per_unit=3)

### Now we work on the heat budget!

