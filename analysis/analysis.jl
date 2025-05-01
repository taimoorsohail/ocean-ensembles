using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2

# output_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/outputs/")
# figdir = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/")

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles-2/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles-2/figures/")

variables_basins = ["_global_", "_atl_", "_ipac_"]
variables_diags = ["zonal", "depth", "tot"]
variables_tracers = ["T", "S", "dV"]
variables_velocities =  ["u", "v", "w"]
variables_vel_volumes = ["dV_u", "dV_v", "dV_w"]
variables_fluxes = ["latent_heat", "sensible_heat", "water_vapor", "x_momentum", "y_momentum"]

tracercontent_vars = vec([tracer * basin * diag for tracer in variables_tracers,
                         basin in variables_basins,
                         diag in variables_diags])

transport_vars = vcat(
    vec([velocity * basin * diag for velocity in variables_velocities,
                                 basin in variables_basins,
                                 diag in variables_diags]),
    vec([volume * basin * diag for volume in variables_vel_volumes,
                               basin in variables_basins,
                               diag in variables_diags])
)

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
OTC = create_dict(tracercontent_vars, output_path * "ocean_tracer_content_RYF1deg.jld2")

@info "I am loading the mass transport" 
masstrans = create_dict(transport_vars, output_path * "mass_transport_RYF1deg.jld2")

@info "I am loading the fluxes" 
fluxes = create_dict(variables_fluxes, output_path * "fluxes_RYF1deg.jld2")


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

save(figdir * "surface_fluxes.png", fig, px_per_unit=3)

# Define the order you want for tracers and basins
tracers = ["T", "S", "dV"]
velocities = ["u", "v", "w"]
basins = ["global", "atl", "ipac"]
minv = [-1,34,1e6]
maxv = [31,38,1e18]

function OTC_visualisation(var, tracers, basins; minv, maxv, diagnostic = "integrate", space = "zonal", dir = "./", time = "last")
    fig = Figure(size = (1600, 800))

    # Loop through and place plots accordingly
    for (row, basin) in enumerate(basins)
        for (col, tracer) in enumerate(tracers)
            key_dv = "dV_$(basin)_$(space)"
            key = "$(tracer)_$(basin)_$(space)"  # Compose the key
            if space == "zonal"
                ax = Axis(fig[row, 2col-1], title=String(key))
            else
                ax = Axis(fig[row, col], title=String(key))
            end

            if time == "last"
                time_slice = lastindex(var[key].times)
            else
                time_slice = Integer(time)
            end
            
            if diagnostic == "integrate"
                data = interior(var[key][time_slice])
            elseif diagnostic == "average"
                data = interior(var[key][time_slice])./(var[key_dv][time_slice])
            end

            if space == "tot"
                if diagnostic == "integrate"
                    data = interior(var[key])
                elseif diagnostic == "average"
                    data = interior(var[key])./(var[key_dv])
                end
            end

            if space == "zonal"
                heatmap!(ax, view(data, 1, :, :), colormap=:viridis, colorrange = (minv[col], maxv[col]))
                Colorbar(fig[row, 2col], label="Units", width=15)
            elseif space == "depth"
                lines!(ax, view(data, 1, 1, :))
            elseif space == "tot"
                lines!(ax, view(data, 1, 1, 1, :))
            else
                throw(ArgumentError("space must be one of zonal, depth or total"))
            end

        end
    end

    title = string("Global 1 degree ocean simulation after ",
                            prettytime(times[time_slice] - times[1]))

    fig[0, :] = Label(fig, "Global 1° ocean simulation after $(prettytime(times[time_slice] - times[1]))", fontsize=24)

    save(dir*"$(space)_$(diagnostic)_tracer_content.png", fig, px_per_unit=3)
end 

OTC_visualisation(OTC, tracers, basins; minv, maxv, diagnostic = "integrate", space = "zonal", dir = figdir, time = "last")
OTC_visualisation(OTC, tracers, basins; minv, maxv, diagnostic = "integrate", space = "depth", dir = figdir, time = "last")
OTC_visualisation(OTC, tracers, basins; minv, maxv, diagnostic = "integrate", space = "tot", dir = figdir, time = "last")
OTC_visualisation(OTC, tracers, basins; minv, maxv, diagnostic = "average", space = "zonal", dir = figdir, time = "last")
OTC_visualisation(OTC, tracers, basins; minv, maxv, diagnostic = "average", space = "depth", dir = figdir, time = "last")
OTC_visualisation(OTC, tracers, basins; minv, maxv, diagnostic = "average", space = "tot", dir = figdir, time = "last")

function masstrans_visualisation(var, tracers, basins; minv, maxv, diagnostic = "integrate", space = "zonal", dir = "./", time = "last")
    fig = Figure(size = (1600, 800))

    # Loop through and place plots accordingly
    for (row, basin) in enumerate(basins)
        for (col, tracer) in enumerate(tracers)
            key_dv = "dV_$(tracer)_$(basin)_$(space)"
            key = "$(tracer)_$(basin)_$(space)"  # Compose the key
            if space == "zonal"
                ax = Axis(fig[row, 2col-1], title=String(key))
            else
                ax = Axis(fig[row, col], title=String(key))
            end

            if time == "last"
                time_slice = lastindex(var[key].times)
            else
                time_slice = Integer(time)
            end
            
            if diagnostic == "integrate"
                data = interior(var[key][time_slice])
            elseif diagnostic == "average"
                data = interior(var[key][time_slice])./(var[key_dv][time_slice])
            end

            if space == "tot"
                if diagnostic == "integrate"
                    data = interior(var[key])
                elseif diagnostic == "average"
                    data = interior(var[key])./(var[key_dv])
                end
            end

            if space == "zonal"
                heatmap!(ax, view(data, 1, :, :), colormap=:viridis, colorrange = (minv[col], maxv[col]))
                Colorbar(fig[row, 2col], label="Units", width=15)
            elseif space == "depth"
                lines!(ax, view(data, 1, 1, :))
            elseif space == "tot"
                lines!(ax, view(data, 1, 1, 1, :))
            else
                throw(ArgumentError("space must be one of zonal, depth or total"))
            end

        end
    end

    title = string("Global 1 degree ocean simulation after ",
                            prettytime(times[time_slice] - times[1]))

    fig[0, :] = Label(fig, "Global 1° ocean simulation after $(prettytime(times[time_slice] - times[1]))", fontsize=24)

    save(dir*"$(space)_$(diagnostic)_voltrans.png", fig, px_per_unit=3)
end

masstrans_visualisation(masstrans, velocities, basins; minv, maxv, diagnostic = "integrate", space = "zonal", dir = figdir, time = "last")
masstrans_visualisation(masstrans, velocities, basins; minv, maxv, diagnostic = "integrate", space = "depth", dir = figdir, time = "last")
masstrans_visualisation(masstrans, velocities, basins; minv, maxv, diagnostic = "integrate", space = "tot", dir = figdir, time = "last")
masstrans_visualisation(masstrans, velocities, basins; minv, maxv, diagnostic = "average", space = "zonal", dir = figdir, time = "last")
masstrans_visualisation(masstrans, velocities, basins; minv, maxv, diagnostic = "average", space = "depth", dir = figdir, time = "last")
masstrans_visualisation(masstrans, velocities, basins; minv, maxv, diagnostic = "average", space = "tot", dir = figdir, time = "last")

# fig = Figure(size = (1600, 800))

# # Define the order you want for tracers and basins
# tracers = ["T", "S", "dV"]
# basins = ["global", "atl", "ipac"]
# minv = [-1,1e16,1e6]
# maxv = [31,5e18,1e18]

# # Loop through and place plots accordingly
# for (row, basin) in enumerate(basins)
#     for (col, tracer) in enumerate(tracers)
#         key_dv = "dV_$(basin)_depth"
#         key = "$(tracer)_$(basin)_depth"  # Compose the key
#         ax = Axis(fig[row, col], title=String(key))
#         time_slice = lastindex(OTC[key].times)
#         data = interior(OTC[key][time_slice]/OTC[key_dv][time_slice])
#         lines!(ax, view(data, 1, 1, :), colormap=:viridis)#, colorrange = (minv[col], maxv[col]))
#     end
# end

# title = string("Global 1 degree ocean simulation after ",
#                          prettytime(times[time_slice] - times[1]))

# fig[0, :] = Label(fig, "Global 1° ocean simulation after $(prettytime(times[time_slice] - times[1]))", fontsize=24)                                 

# save("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/depth_tracer_content.png", fig, px_per_unit=3)

# fig = Figure(size = (1600, 800))

# # Define the order you want for tracers and basins
# tracers = ["T", "S", "dV"]
# basins = ["global", "atl", "ipac"]
# minv = [-1,1e16,1e6]
# maxv = [31,5e18,1e18]

# # Loop through and place plots accordingly
# for (row, basin) in enumerate(basins)
#     for (col, tracer) in enumerate(tracers)
#         key_dv = "dV_$(basin)_tot"
#         key = "$(tracer)_$(basin)_tot"  # Compose the key
#         ax = Axis(fig[row, col], title=String(key))
#         data = interior(OTC[key])./interior(OTC[key_dv])
#         lines!(ax, view(data, 1, 1, 1, :), colormap=:viridis)#, colorrange = (minv[col], maxv[col]))
#     end
# end

# title = string("Global 1 degree ocean simulation after ",
#                          prettytime(times[time_slice] - times[1]))

# fig[0, :] = Label(fig, "Global 1° ocean simulation after $(prettytime(times[time_slice] - times[1]))", fontsize=24)

# save("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/global_tracer_content.png", fig, px_per_unit=3)

# fig = Figure(size = (1600, 800))

# # Define the order you want for tracers and basins
# tracers = ["u", "v", "w", "dV"]
# basins = ["global", "atl", "ipac"]
# minv = [-1,1e16,1e6]
# maxv = [31,5e18,1e18]

# # Loop through and place plots accordingly
# for (row, basin) in enumerate(basins)
#     for (col, tracer) in enumerate(tracers)
#         key_dv = "dV_$(basin)_zonal"
#         key = "$(tracer)_$(basin)_zonal"  # Compose the key
#         ax = Axis(fig[row, 2col-1], title=String(key))
#         time_slice = lastindex(masstrans[key].times)
#         data = interior(masstrans[key][time_slice]/masstrans[key_dv][time_slice])
#         heatmap!(ax, view(data, 1, :, :), colormap=:viridis)#, colorrange = (minv[col], maxv[col]))
#         Colorbar(fig[row, 2col], label="Units", width=15)
#     end
# end

# title = string("Global 1 degree ocean simulation after ",
#                          prettytime(times[time_slice] - times[1]))

# fig[0, :] = Label(fig, "Global 1° ocean simulation after $(prettytime(times[time_slice] - times[1]))", fontsize=24)

# save("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/zonal_masstrans.png", fig, px_per_unit=3)


