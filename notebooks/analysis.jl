using CairoMakie, GLMakie
using Oceananigans  # From local
using Statistics
using JLD2

experiment_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/experiments/")

variables_basins = ["_global_", "_atl_", "_ipac_"]
variables_tracers = ["T", "S", "u", "v", "w", "dV", "dV_u", "dV_v", "dV_w"]
variables_diags = ["zonal", "depth", "tot"]
variables_fluxes = ["latent_heat", "sensible_heat", "water_vapor", "x_momentum", "y_momentum"]
# Create all combinations of tracer, basin, and diag
variable_names = [tracer * basin * diag for tracer in variables_tracers,
                                          basin in variables_basins,
                                          diag in variables_diags]

println(variable_names)

function create_dict(vars, path)
    dicts = Dict()
    for var in vars
        try
            # Surface
            dicts[var] = FieldTimeSeries(path, var)
        catch e
            if e isa KeyError
                @warn "Skipping variable $var: Key not found in file."
            else
                rethrow(e)
            end
        end
    end
    return dicts
end

@info "I am loading the otc" 
OTC = create_dict(variable_names, experiment_path * "ocean_tracer_content.jld2")

@info "I am loading the mass transport" 
masstrans = create_dict(variable_names, experiment_path * "mass_transport.jld2")

@info "I am loading the fluxes" 
fluxes = create_dict(variables_fluxes, experiment_path * "fluxes.jld2")


fig = Figure(size = (1600, 800))
# axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

time_slice = lastindex(fluxes["latent_heat"].times)

times = fluxes["latent_heat"].times

title = string("Global 1 degree ocean simulation after ",
                         prettytime(times[time_slice] - times[1]))


axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
hm = heatmap!(axs, view(interior(fluxes["latent_heat"][time_slice]), :, :, 1), colorrange = (-500,500), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[1, 2], hm, label = "Latent Heat (W/m2)")

axs = Axis(fig[1, 3], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
hm = heatmap!(axs, view(interior(fluxes["sensible_heat"][time_slice]), :, :, 1), colorrange = (-500,500), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[1, 4], hm, label = "Sensible Heat (W/m2)")
Label(fig[0, :], title)

save("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/surface_fluxes.png", fig, px_per_unit=3)

fig = Figure(size = (1600, 800))

# Define the order you want for tracers and basins
tracers = ["T", "S", "dV"]
basins = ["global", "atl", "ipac"]
minv = [-1,1e16,1e6]
maxv = [31,5e18,1e18]

# Loop through and place plots accordingly
for (row, basin) in enumerate(basins)
    for (col, tracer) in enumerate(tracers)
        key_dv = "dV_$(basin)_zonal"
        key = "$(tracer)_$(basin)_zonal"  # Compose the key
        ax = Axis(fig[row, 2col-1], title=String(key))
        time_slice = lastindex(OTC[key].times)
        data = interior(OTC[key][time_slice]/OTC[key_dv][time_slice])
        heatmap!(ax, view(data, 1, :, :), colormap=:viridis)#, colorrange = (minv[col], maxv[col]))
        Colorbar(fig[row, 2col], label="Units", width=15)
    end
end

title = string("Global 1 degree ocean simulation after ",
                         prettytime(times[time_slice] - times[1]))

fig[0, :] = Label(fig, "Global 1° ocean simulation after $(prettytime(times[time_slice] - times[1]))", fontsize=24)

save("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/zonal_tracer_content.png", fig, px_per_unit=3)

fig = Figure(size = (1600, 800))

# Define the order you want for tracers and basins
tracers = ["T", "S", "dV"]
basins = ["global", "atl", "ipac"]
minv = [-1,1e16,1e6]
maxv = [31,5e18,1e18]

# Loop through and place plots accordingly
for (row, basin) in enumerate(basins)
    for (col, tracer) in enumerate(tracers)
        key_dv = "dV_$(basin)_depth"
        key = "$(tracer)_$(basin)_depth"  # Compose the key
        ax = Axis(fig[row, col], title=String(key))
        time_slice = lastindex(OTC[key].times)
        data = interior(OTC[key][time_slice]/OTC[key_dv][time_slice])
        lines!(ax, view(data, 1, 1, :), colormap=:viridis)#, colorrange = (minv[col], maxv[col]))
    end
end

title = string("Global 1 degree ocean simulation after ",
                         prettytime(times[time_slice] - times[1]))

fig[0, :] = Label(fig, "Global 1° ocean simulation after $(prettytime(times[time_slice] - times[1]))", fontsize=24)                                 

save("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/depth_tracer_content.png", fig, px_per_unit=3)

fig = Figure(size = (1600, 800))

# Define the order you want for tracers and basins
tracers = ["T", "S", "dV"]
basins = ["global", "atl", "ipac"]
minv = [-1,1e16,1e6]
maxv = [31,5e18,1e18]

# Loop through and place plots accordingly
for (row, basin) in enumerate(basins)
    for (col, tracer) in enumerate(tracers)
        key_dv = "dV_$(basin)_tot"
        key = "$(tracer)_$(basin)_tot"  # Compose the key
        ax = Axis(fig[row, col], title=String(key))
        data = interior(OTC[key])./interior(OTC[key_dv])
        lines!(ax, view(data, 1, 1, 1, :), colormap=:viridis)#, colorrange = (minv[col], maxv[col]))
    end
end

title = string("Global 1 degree ocean simulation after ",
                         prettytime(times[time_slice] - times[1]))

fig[0, :] = Label(fig, "Global 1° ocean simulation after $(prettytime(times[time_slice] - times[1]))", fontsize=24)

save("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/global_tracer_content.png", fig, px_per_unit=3)

fig = Figure(size = (1600, 800))

# Define the order you want for tracers and basins
tracers = ["u", "v", "w", "dV"]
basins = ["global", "atl", "ipac"]
minv = [-1,1e16,1e6]
maxv = [31,5e18,1e18]

# Loop through and place plots accordingly
for (row, basin) in enumerate(basins)
    for (col, tracer) in enumerate(tracers)
        key_dv = "dV_$(basin)_zonal"
        key = "$(tracer)_$(basin)_zonal"  # Compose the key
        ax = Axis(fig[row, 2col-1], title=String(key))
        time_slice = lastindex(masstrans[key].times)
        data = interior(masstrans[key][time_slice]/masstrans[key_dv][time_slice])
        heatmap!(ax, view(data, 1, :, :), colormap=:viridis)#, colorrange = (minv[col], maxv[col]))
        Colorbar(fig[row, 2col], label="Units", width=15)
    end
end

title = string("Global 1 degree ocean simulation after ",
                         prettytime(times[time_slice] - times[1]))

fig[0, :] = Label(fig, "Global 1° ocean simulation after $(prettytime(times[time_slice] - times[1]))", fontsize=24)

save("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/zonal_masstrans.png", fig, px_per_unit=3)


