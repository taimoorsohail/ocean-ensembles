# using CairoMakie
# using Oceananigans  # From local
# using Statistics
# using JLD2
using OceanEnsembles

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")
prefix = output_path * "global_surface_fields_threedeg"

combine_ranks(prefix, prefix; remove_split_files = true, gridtype = "LatitudeLongitudeGrid")

# variables_tracers = ["T", "S"]
# variables_velocities =  ["u", "v", "w"]
# variables_fluxes = ["latent_heat", "sensible_heat", "water_vapor", "x_momentum", "y_momentum", 
#                     "upwelling_longwave", "downwelling_longwave", "downwelling_shortwave"]
# variables_surface = vcat(variables_tracers, variables_velocities)

# function create_dict(vars, path)
#     dicts = Dict()
#     for var in vars
#         try
#             # Surface
#             dicts[var] = FieldTimeSeries(path, var)
#         catch e
#             if e isa KeyError
#                 @warn "Skipping variable $var: Key not found in file."
#             else
#                 rethrow(e)
#             end
#         end
#     end
#     return dicts
# end

# @info "I am loading the fluxes" 
# # fluxes = create_dict(variables_fluxes, output_path * filename_flux)
# @info "I am loading the surface properties" 
# surface = create_dict(variables_surface, output_path * filename_surf)
# # fluxes2 = create_dict(variables_fluxes, output_path * filename_flux2)
# # @info "I am loading the surface properties" 
# # surface2 = create_dict(variables_surface, output_path * filename_surf2)

# times = surface["T"].times
# time_slice = lastindex(surface["T"].times)


# ###### SURFACE FLUXES ######

# # axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

# time_slice = lastindex(fluxes2["latent_heat"].times)

# times = fluxes2["latent_heat"].times
# trange = (time_slice-75):time_slice
# surface_Lflux_1 = mean(view(fluxes["latent_heat"], :, :, 1, trange), dims = 3)
# surface_Sflux_1 = mean(view(fluxes["sensible_heat"], :, :, 1, trange), dims = 3)
# surface_ULWflux_1 = mean(view(fluxes["upwelling_longwave"], :, :, 1, trange), dims = 3)
# surface_DLWflux_1 = mean(view(fluxes["downwelling_longwave"], :, :, 1, trange), dims = 3)
# surface_DSWflux_1 = mean(view(fluxes["downwelling_shortwave"], :, :, 1, trange), dims = 3)

# surface_Lflux_2 = mean(view(fluxes2["latent_heat"], :, :, 1, trange), dims = 3)
# surface_Sflux_2 = mean(view(fluxes2["sensible_heat"], :, :, 1, trange), dims = 3)
# surface_ULWflux_2 = mean(view(fluxes2["upwelling_longwave"], :, :, 1, trange), dims = 3)
# surface_DLWflux_2 = mean(view(fluxes2["downwelling_longwave"], :, :, 1, trange), dims = 3)
# surface_DSWflux_2 = mean(view(fluxes2["downwelling_shortwave"], :, :, 1, trange), dims = 3)

# fig = Figure(size = (1600, 1600))

# title = string("Global 1 degree ocean simulation averaged over final ",
#                     prettytime(sum(diff(times[trange]))))

# axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_Lflux_1), :, :, 1), colorrange = (-240,240), colormap = :bwr, nan_color=:lightgray)
# contour!(axs, view((surface_Lflux_1), :, :, 1), levels = [75,150], color= :black)
# Colorbar(fig[1, 2], hm, label = "Latent Heat (W/m2)")

# axs = Axis(fig[1, 3], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_Sflux_1), :, :, 1), colorrange = (-60,60), colormap = :bwr, nan_color=:lightgray)
# contour!(axs, view((surface_Sflux_1), :, :, 1), levels = [20,40], color= :black)
# Colorbar(fig[1, 4], hm, label = "Sensible Heat (W/m2)")
# Label(fig[0, :], title)

# axs = Axis(fig[2, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_ULWflux_1), :, :, 1), colorrange = (-500,500), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[2, 2], hm, label = "Upward Longwave Radiation (W/m2)")

# axs = Axis(fig[2, 3], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_DLWflux_1), :, :, 1), colorrange = (-500,500), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[2, 4], hm, label = "Downward Longwave Radiation (W/m2)")
# Label(fig[0, :], title)

# axs = Axis(fig[3, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_DSWflux_1), :, :, 1), colorrange = (-300,300), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[3, 2], hm, label = "Downward Shortwave Radiation (W/m2)")
# Label(fig[0, :], title)

# save(figdir * "surface_fluxes_" * suffix * ".png", fig, px_per_unit=3)

# fig = Figure(size = (1600, 1600))

# title = string("Global 1 degree ocean simulation averaged over final ",
#                     prettytime(sum(diff(times[trange]))))

# axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_Lflux_2), :, :, 1), colorrange = (-240,240), colormap = :bwr, nan_color=:lightgray)
# contour!(axs, view((surface_Lflux_2), :, :, 1), levels = [75,150], color= :black)
# Colorbar(fig[1, 2], hm, label = "Latent Heat (W/m2)")

# axs = Axis(fig[1, 3], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_Sflux_2), :, :, 1), colorrange = (-60,60), colormap = :bwr, nan_color=:lightgray)
# contour!(axs, view((surface_Sflux_2), :, :, 1), levels = [20,40], color= :black)
# Colorbar(fig[1, 4], hm, label = "Sensible Heat (W/m2)")
# Label(fig[0, :], title)

# axs = Axis(fig[2, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_ULWflux_2), :, :, 1), colorrange = (-500,500), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[2, 2], hm, label = "Upward Longwave Radiation (W/m2)")

# axs = Axis(fig[2, 3], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_DLWflux_2), :, :, 1), colorrange = (-500,500), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[2, 4], hm, label = "Downward Longwave Radiation (W/m2)")
# Label(fig[0, :], title)

# axs = Axis(fig[3, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_DSWflux_2), :, :, 1), colorrange = (-300,300), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[3, 2], hm, label = "Downward Shortwave Radiation (W/m2)")
# Label(fig[0, :], title)

# save(figdir * "surface_fluxes_" * suffix2 * ".png", fig, px_per_unit=3)

# fig = Figure(size = (1600, 1600))

# title = string("Global 1 degree ocean simulation averaged over final ",
#                          prettytime(sum(diff(times[trange]))))

# axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_Lflux_1-surface_Lflux_2), :, :, 1), colorrange = (-20,20), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[1, 2], hm, label = "Latent Heat \nFixed minus zstar (W/m2)")

# axs = Axis(fig[1, 3], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_Sflux_1-surface_Sflux_2), :, :, 1), colorrange = (-20,20), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[1, 4], hm, label = "Sensible Heat \nFixed minus zstar (W/m2)")
# Label(fig[0, :], title)

# axs = Axis(fig[2, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_ULWflux_1-surface_ULWflux_2), :, :, 1), colorrange = (-20,20), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[2, 2], hm, label = "Upward Longwave Radiation \nFixed minus zstar (W/m2)")

# axs = Axis(fig[2, 3], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_DLWflux_1-surface_DLWflux_2), :, :, 1), colorrange = (-20,20), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[2, 4], hm, label = "Downward Longwave Radiation \nFixed minus zstar (W/m2)")
# Label(fig[0, :], title)

# axs = Axis(fig[3, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view((surface_DSWflux_1-surface_DSWflux_2), :, :, 1), colorrange = (-20,20), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[3, 2], hm, label = "Downward Shortwave Radiation \nFixed minus zstar (W/m2)")
# Label(fig[0, :], title)

# save(figdir * "surface_fluxes_" * suffix * "_diff.png", fig, px_per_unit=3)

# ##################################################################

# using Oceananigans.Operators: Az
# LH_W = zeros(length(times))  # Ensure the array is 1D with the correct size
# SH_W = zeros(length(times))  # Ensure the array is 1D with the correct size
# LH_W2 = zeros(length(times))  # Ensure the array is 1D with the correct size
# SH_W2 = zeros(length(times))  # Ensure the array is 1D with the correct size

# for time_step in eachindex(times)  # Proper way to iterate over indices
#     LH_W[time_step] = sum(interior(fluxes["latent_heat"][time_step]*Az))
#     SH_W[time_step] = sum(interior(fluxes["sensible_heat"][time_step]*Az))
#     LH_W2[time_step] = sum(interior(fluxes2["latent_heat"][time_step]*Az))
#     SH_W2[time_step] = sum(interior(fluxes2["sensible_heat"][time_step]*Az))
# end

# mxtemp = zeros(length(times))  # Ensure the array is 1D with the correct size
# mntemp = zeros(length(times))  # Ensure the array is 1D with the correct size
# hightmp_counter = zeros(length(times))
# mxtemp2 = zeros(length(times))  # Ensure the array is 1D with the correct size
# mntemp2 = zeros(length(times))  # Ensure the array is 1D with the correct size
# hightmp_counter2 = zeros(length(times))

# for time_step in eachindex(times)  # Proper way to iterate over indices
#     mxtemp[time_step] = maximum(surface["T"][time_step])  # Ignore NaNs
#     mntemp[time_step] = minimum(surface["T"][time_step])  # Ignore NaNs
#     T_array = interior(surface["T"][time_step])
#     # Extract the interior data
#     interior_data = interior(surface["T"][time_step])

#     # Find the (i, j) index of the maximum temperature in the interior
#     max_index = argmax(interior_data)

#     mxtemp2[time_step] = maximum(surface2["T"][time_step])  # Ignore NaNs
#     mntemp2[time_step] = minimum(surface2["T"][time_step])  # Ignore NaNs
#     T_array2 = interior(surface2["T"][time_step])
#     # Extract the interior data
#     interior_data2 = interior(surface2["T"][time_step])

#     # Find the (i, j) index of the maximum temperature in the interior
#     max_index2 = argmax(interior_data2)
# end

# fig = Figure(size = (1200, 800))
# ax1 = Axis(fig[1, 1], xlabel="Years", ylabel="Latent Heat [W]", title="Latent Heat")
# ax2 = Axis(fig[1, 2], xlabel="Years", ylabel="Sensible Heat [W]", title="Sensible Heat")
# ax3 = Axis(fig[2,1], xlabel="Years", ylabel="Temperature", title="Surface Temperature Extrema")

# lines!(ax1, times/(3600*24*365.25), LH_W, color=:black, label = suffix)
# lines!(ax2, times/(3600*24*365.25), SH_W, color=:black)
# lines!(ax3, times/(3600*24*365.25), mxtemp, color=:red)
# lines!(ax1, times/(3600*24*365.25), LH_W2, color=:black, label = suffix2, linestyle = :dash)
# lines!(ax2, times/(3600*24*365.25), SH_W2, color=:black, linestyle = :dash)
# lines!(ax3, times/(3600*24*365.25), mxtemp2, color=:red, linestyle = :dash)
# axislegend(ax1, position = :rb)

# dt_change = 20 / 365.25  # 20 days in years

# vlines!(ax1, [dt_change], color=:black, linestyle=:dash, label = "Increasing dt")
# vlines!(ax2, [dt_change], color=:black, linestyle=:dash)
# vlines!(ax3, [dt_change], color=:black, linestyle=:dash)

# save(figdir * "flux_issue_" * suffix * ".png", fig, px_per_unit=3)

# ###### SURFACE FIELDS ######

# surface_temp_1 = mean(interior(surface["T"], :, :, 1, trange), dims = 3)
# surface_salt_1 = mean(interior(surface["S"], :, :, 1, trange), dims = 3)
# surface_u_1 = mean(interior(surface["u"], :, :, 1, trange), dims = 3)
# surface_v_1 = mean(interior(surface["v"], :, :, 1, trange), dims = 3)

# surface_temp_2 = mean(interior(surface2["T"], :, :, 1, trange), dims = 3)
# surface_salt_2 = mean(interior(surface2["S"], :, :, 1, trange), dims = 3)
# surface_u_2 = mean(interior(surface2["u"], :, :, 1, trange), dims = 3)
# surface_v_2 = mean(interior(surface2["v"], :, :, 1, trange), dims = 3)

# fig = Figure(size = (1200, 800))
# # axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

# title = string("Global 1 degree ocean simulation averaged over final ",
#                          prettytime(sum(diff(times[trange]))))

# axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view(surface_temp_1, :, :, 1), colorrange = (-1,40), colormap = :bwr, nan_color=:lightgray)
# contour!(axs, view(surface_temp_1, :, :, 1), levels=[35.0], color = :black)

# Colorbar(fig[1, 2], hm, label = "Temperature (ᵒC)")

# axs = Axis(fig[2, 1], xlabel="Longitude (deg)", ylabel="Depth (m)")
# hm = heatmap!(axs, view(surface_u_1, :, :, 1) , colorrange = (-1,1), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[2, 2], hm, label = "u (m/s)")

# axs = Axis(fig[2, 3], xlabel="Longitude (deg)", ylabel="Depth (m)")
# hm = heatmap!(axs, view(surface_v_1, :, :, 1) , colorrange = (-1,1), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[2, 4], hm, label = "v (m/s)")

# axs = Axis(fig[1, 3], xlabel="Longitude (deg)", ylabel="Depth (m)")
# hm = heatmap!(axs, view(surface_salt_1, :, :, 1) , colorrange = (33, 38), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[1, 4], hm, label = "Salinity (unitless)")


# Label(fig[0, :], title)
# save(figdir * "surface_fields_" * suffix * ".png", fig, px_per_unit=3)                              

# fig = Figure(size = (1200, 800))
# # axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

# title = string("Global 1 degree ocean simulation averaged over final ",
#                          prettytime(sum(diff(times[trange]))))

# axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view(surface_temp_2, :, :, 1), colorrange = (-1,40), colormap = :bwr, nan_color=:lightgray)
# contour!(axs, view(surface_temp_2, :, :, 1), levels=[35.0], color = :black)

# Colorbar(fig[1, 2], hm, label = "Temperature (ᵒC)")

# axs = Axis(fig[2, 1], xlabel="Longitude (deg)", ylabel="Depth (m)")
# hm = heatmap!(axs, view(surface_u_2, :, :, 1) , colorrange = (-1,1), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[2, 2], hm, label = "u (m/s)")

# axs = Axis(fig[2, 3], xlabel="Longitude (deg)", ylabel="Depth (m)")
# hm = heatmap!(axs, view(surface_v_2, :, :, 1) , colorrange = (-1,1), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[2, 4], hm, label = "v (m/s)")

# axs = Axis(fig[1, 3], xlabel="Longitude (deg)", ylabel="Depth (m)")
# hm = heatmap!(axs, view(surface_salt_2, :, :, 1) , colorrange = (33, 38), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[1, 4], hm, label = "Salinity (unitless)")


# Label(fig[0, :], title)
# save(figdir * "surface_fields_" * suffix2 * ".png", fig, px_per_unit=3)                              

# fig = Figure(size = (1200, 800))
# # axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

# title = string("Global 1 degree ocean simulation averaged over final ",
#                          prettytime(sum(diff(times[trange]))))

# axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
# hm = heatmap!(axs, view(surface_temp_1-surface_temp_2, :, :, 1), colorrange = (-5,5), colormap = :bwr, nan_color=:lightgray)
# contour!(axs, view(surface_temp_1-surface_temp_2, :, :, 1), levels=range(-5,5), color = :black)

# Colorbar(fig[1, 2], hm, label = "Temperature (ᵒC)")

# axs = Axis(fig[2, 1], xlabel="Longitude (deg)", ylabel="Depth (m)")
# hm = heatmap!(axs, view(surface_u_1-surface_u_2, :, :, 1) , colorrange = (-0.05,0.05), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[2, 2], hm, label = "u (m/s)")

# axs = Axis(fig[2, 3], xlabel="Longitude (deg)", ylabel="Depth (m)")
# hm = heatmap!(axs, view(surface_v_1-surface_v_2, :, :, 1) , colorrange = (-.05,.05), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[2, 4], hm, label = "v (m/s)")

# axs = Axis(fig[1, 3], xlabel="Longitude (deg)", ylabel="Depth (m)")
# hm = heatmap!(axs, view(surface_salt_1-surface_salt_2, :, :, 1) , colorrange = (-1,1), colormap = :bwr, nan_color=:lightgray)
# Colorbar(fig[1, 4], hm, label = "Salinity (unitless)")


# Label(fig[0, :], title)
# save(figdir * "surface_fields_" * suffix * "_diff.png", fig, px_per_unit=3)                              