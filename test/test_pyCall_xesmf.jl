
using GLMakie 

using OceanEnsembles
using Oceananigans

destination_grid   = LatitudeLongitudeGrid(size=(360, 180, 20), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

source_grid = TripolarGrid(CPU();
                           size = (360,180, 20),
                           z =  (0,1),
                           halo = (6, 6, 3),
                           first_pole_longitude = 70,
                           north_poles_latitude = 55,
                           southernmost_latitude = -80)

source_field = Field{Center, Center, Center}(source_grid)

f(λ, φ, z) = φ < 0 ? 0 : λ/λ * 1
Oceananigans.set!(source_field, f)
# Test the regridding function
dst, W = regrid_tracers!(source_field, destination_grid; method = "conservative", output_weights = true)

# Plotting the regridded data vs the source data
fig1 = Figure()
ax1 = Axis(fig1[1, 1], title="Source Field", xlabel="i", ylabel="j")
ax2 = Axis(fig1[3, 1], title="Destination Field", xlabel="Longitude", ylabel="Latitude")

heatmap!(ax1, collect(interior(source_field))[:,:,19], colormap=:viridis, colorrange=(minimum(source_field), maximum(source_field)))
Colorbar(fig1[2,1], label = "Tracer", vertical=false, colorrange=(minimum(source_field), maximum(source_field)))

heatmap!(ax2, collect(interior(dst))[:,:,19], colormap=:viridis, colorrange=(minimum(dst), maximum(dst)))
Colorbar(fig1[4,1], label = "Tracer", vertical=false, colorrange=(minimum(dst), maximum(dst)))

display(fig1)

# fig = Figure()
# ax1 = Axis(fig[1, 1], title="Source Areas", xlabel="Point #", ylabel="Area")
# ax2 = Axis(fig[3, 1], title="Dest. Areas", xlabel="Point #", ylabel="Area")

# heatmap!(ax1, reshape(row_sums, 360,180), colormap=:viridis, colorrange=(0, maximum(row_sums)))
# Colorbar(fig[2,1], label = "Source Temperature", vertical=false, colorrange=(0, maximum(row_sums)))

# heatmap!(ax2, (reshape(W_sums, 360,180))  , colorrange=(0, maximum(W_sums)))
# Colorbar(fig[4,1], label = "Dest. Temperature", vertical=false, colorrange=(0, maximum(W_sums)))

# display(fig)
