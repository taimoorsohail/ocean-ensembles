using Oceananigans.Fields: x_integral_regrid!, y_integral_regrid!, z_integral_regrid!
using ConservativeRegridding
using Oceananigans
using Statistics 
using GLMakie 

fine_grid   = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))


# coarse_grid = LatitudeLongitudeGrid(size=(90, 45, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

coarse_grid = TripolarGrid(CPU();
                           size = (360,180, 1),
                           z =  (0,1),
                           halo = (6, 6, 3),
                           first_pole_longitude = 70,
                           north_poles_latitude = 55)

dst = Field{Center, Center, Nothing}(fine_grid)
src = Field{Center, Center, Nothing}(coarse_grid)

regridder = ConservativeRegridding.Regridder(dst, src)

set!(src, (x, y) -> rand())

new_vals = reshape(ConservativeRegridding.regrid!(dst, regridder, src), 360, 180)

@show mean(dst)

@show mean(src)

# Plotting the regridded data
fig = Figure()
ax1 = Axis(fig[1, 1], title="Dest. Areas", xlabel="Point #", ylabel="Area")
ax2 = Axis(fig[1, 2], title="Source Areas", xlabel="Point #", ylabel="Area")

scatter!(ax1, regridder.dst_areas)
scatter!(ax2, regridder.src_areas)
display(fig)