using PyCall
using Oceananigans
using Statistics 
using GLMakie 

np = pyimport("numpy")
xesmf = pyimport("xesmf")
xr = pyimport("xarray")

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

Oceananigans.set!(src, (x, y) -> rand())

src_np = np.squeeze(np.array(collect(interior(src))))
dst_np = np.squeeze(np.array(collect(interior(dst))))

# Extract Centers

src_lats = src.grid.φᶜᶜᵃ[1:end - src.grid.Hx, 1:end - src.grid.Hy]
src_lons = src.grid.λᶜᶜᵃ[1:end - src.grid.Hx, 1:end - src.grid.Hy]

dst_lats = dst.grid.φᵃᶜᵃ[1:end - dst.grid.Hy]
dst_lons = dst.grid.λᶜᵃᵃ[1:end - dst.grid.Hx]

# Extract corners

src_lats_b = src.grid.φᶠᶠᵃ[1:end - src.grid.Hx+1, 1:end - src.grid.Hy+1]
src_lons_b = src.grid.λᶠᶠᵃ[1:end - src.grid.Hx+1, 1:end - src.grid.Hy+1]

dst_lats_b = dst.grid.φᵃᶠᵃ[1:end - dst.grid.Hy]
dst_lons_b = dst.grid.λᶠᵃᵃ[1:end - dst.grid.Hx]

# Convert to numpy arrays for xesmf
lat_src_np = np.array(src_lats)
lon_src_np = np.array(src_lons)

lat_dst_np = np.array(dst_lats)
lon_dst_np = np.array(dst_lons)

lat_src_b_np = np.array(src_lats_b)
lon_src_b_np = np.array(src_lons_b)

lat_dst_b_np = np.array(dst_lats_b)
lon_dst_b_np = np.array(dst_lons_b)

src_da = xr.DataArray(
    src_np,
    dims=["y", "x"],
    coords= PyObject(Dict(
        "lat" => (["y", "x"], lat_src_np),
        "lon" => (["y", "x"], lon_src_np)
    )),
    name="temperature"
)

src_b_da =  xr.DataArray(
    lat_src_b_np,
    dims=["y_b", "x_b"],
    coords= PyObject(Dict(
        "lat_b" => (["y_b", "x_b"], lat_src_b_np),
        "lon_b" => (["y_b", "x_b"], lon_src_b_np)
    )),
    name="bounds"
)

src_ds = xr.Dataset(
    PyObject(Dict("temperature" => src_da, "bounds" => src_b_da))
)

dst_ds = xr.DataArray(
    dst_np,
    dims=["lon", "lat"],
    coords= PyObject(Dict(
        "lat" => (["lat"], lat_dst_np),
        "lon" => (["lon"], lon_dst_np)    )),
    name="temperature"
)

ds_out = xesmf.util.grid_global(1, 1)

regridder = xesmf.Regridder(src_ds, ds_out, "conservative")

# src_data = np.array(rand(size(lat_src)))

# Perform regridding
dst_data = regridder(src_np)

# Convert back to Julia
dst_data_julia = Array(dst_data)
src_data_julia = Array(src_np)
@show mean(dst_data_julia)
@show mean(src_data_julia)




# regridder = ConservativeRegridding.Regridder(dst, src)

# set!(src, (x, y) -> rand())

# new_vals = reshape(ConservativeRegridding.regrid!(dst, regridder, src), 360, 180)

# @show mean(dst)

# @show mean(src)

# # Plotting the regridded data
# fig = Figure()
# ax1 = Axis(fig[1, 1], title="Dest. Areas", xlabel="Point #", ylabel="Area")
# ax2 = Axis(fig[1, 2], title="Source Areas", xlabel="Point #", ylabel="Area")

# scatter!(ax1, regridder.dst_areas)
# scatter!(ax2, regridder.src_areas)
# display(fig)