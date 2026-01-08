using ClimaOcean
using Oceananigans
using Oceananigans.Operators: Ax, Ay, Az, Δz, Δx, Δy
using Oceananigans.Fields: x_integral_regrid!, y_integral_regrid!, z_integral_regrid!
using ConservativeRegridding
using Statistics

arch = CPU()#Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
Nx, Ny, Nz = 300,300,30#360*6, 180*6, 75
# @info "Defining vertical z faces"
# depth = -6000.0 # Depth of the ocean in meters
# r_faces = ExponentialCoordinate(Nz, depth, 0)
@info "Defining grid"

# grid_tripolar = TripolarGrid(arch;
#                     size = (Nx, Ny, 1),
#                     z =  (0,1),
#                     halo = (6, 6, 3),
#                     first_pole_longitude = 70,
#                     north_poles_latitude = 55)
grid_tripolar = LatitudeLongitudeGrid(arch;
                             size = (Nx+10, Ny+10, 1),
                             halo = (7, 7, 7),
                             z = (0,1),
                             latitude  = (-90,90),
                             longitude = (0, 360))


grid_latlon = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, 1),
                             halo = (7, 7, 7),
                             z = (0,1),
                             latitude  = (-90,90),
                             longitude = (0, 360))

Tgrid = CenterField(grid_tripolar)
Tgrid_ones = CenterField(grid_tripolar)

Tgrid_2 = CenterField(grid_latlon)
Tgrid_2_ones = CenterField(grid_latlon)

function temp_lon_spherical(λ, φ, z)
    #This function should integrate to zero
    return sin(λ * π / 180) + sin(φ * π / 180)
end

set!(Tgrid, temp_lon_spherical)
set!(Tgrid_ones, 1)
set!(Tgrid_2, temp_lon_spherical)
set!(Tgrid_2_ones, 1)

# dst = Tgrid
# src = Tgrid_2

# regridder = ConservativeRegridding.Regridder(dst, src)

# @info "   Testing forward regridding on LatitudeLongitudeGrid..."
# set!(src, (x, y, z) -> rand())
# ConservativeRegridding.regrid!(dst, regridder, src)
# @info mean(dst)
# @info mean(src)

coarse_grid = TripolarGrid(CPU();
                    size = (360,180, 1),
                    z =  (0,1),
                    halo = (6, 6, 3),
                    first_pole_longitude = 70,
                    north_poles_latitude = 55)
fine_grid   = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

dst = Field{Center, Center, Nothing}(fine_grid)
src = Field{Center, Center, Nothing}(coarse_grid)

regridder = ConservativeRegridding.Regridder(dst, src)

set!(src, (x, y) -> rand())
new_vals = reshape(ConservativeRegridding.regrid!(dst, regridder, src), 360,180)

@show mean(dst)
@show mean(src)

#=


function make_vertices_from_grid(fld::Field)
    lons = collect(fld.grid.λᶠᵃᵃ)  # 1D array of longitudes
    lats = collect(fld.grid.φᵃᶠᵃ)  # 1D array of latitudes
    @info size(lons), size(lats)
    vertices = []  # Vector of coordinate arrays

    # For each grid cell, create a polygon by using four corners:
    for j in 1:length(lats)-1, i in 1:length(lons)-1
        coords = [
            (lons[i],     lats[j]),     # lower-left corner
            (lons[i+1],   lats[j]),     # lower-right corner
            (lons[i+1],   lats[j+1]),   # upper-right corner
            (lons[i],     lats[j+1]),   # upper-left corner
            (lons[i],     lats[j])      # close ring (back to lower-left)
        ]
        push!(vertices, coords)
    end

    return vertices
end


dst, src = make_vertices_from_grid(dst_field), make_vertices_from_grid(src_field)

regridder = ConservativeRegridding.Regridder(dst_field, src_field)

@info "   Testing forward regridding on LatitudeLongitudeGrid..."
set!(src_field, (x, y, z) -> rand())

src_arr = copy(parent(src_field)[2:end,:,2])
dst_arr = copy(parent(dst_field)[2:end,:,2])

function flatten_field(array)
    return reshape(array, :, 1)  # shape (ncell, 1)
end

src_arr_flat = flatten_field(src_arr)
dst_arr_flat = flatten_field(dst_arr)

@info size(src_arr_flat)
@info size(dst_arr_flat)

println("regridder.intersections size: ", size(regridder.intersections))
println("src_2d size: ", size(src_arr))
println("dst_2d size: ", size(dst_arr))

@info size(dst_arr_flat), size(src_arr_flat)


# @info "   Testing backward regridding on LatitudeLongitudeGrid..."
# set!(dst, (x, y, z) -> rand())
# ConservativeRegridding.regrid!(src, transpose(regridder), dst)
# @test mean(c2) ≈ mean(c1)





fig = Figure(size = (1600, 800))

axs1 = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
axs2 = Axis(fig[2, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

hm = heatmap!(axs1, interior(Tgrid, :, :, 1))
hightmp_counter2 = heatmap!(axs2, interior(Tgrid_2, :, :, 1))
Colorbar(fig[3,1], hm, label = "Temperature (C)")
display(fig)

Tgrid_int = compute!(Integral(Tgrid))
Tgrid_ones_int = compute!(Integral(Tgrid_ones))
Tgrid_2_int = compute!(Integral(Tgrid_2))
Tgrid_2_ones_int = compute!(Integral(Tgrid_2_ones))

@show Field(Tgrid_int)[1,1,1]
@show Field(Tgrid_int)[1,1,1]/Field(Tgrid_ones_int)[1,1,1] * 100

check_integral1 = Field(Tgrid_int)[1,1,1]/Field(Tgrid_ones_int)[1,1,1] * 100 < 1e-14

@show Field(Tgrid_2_int)[1,1,1]
@show Field(Tgrid_2_int)[1,1,1]/Field(Tgrid_2_ones_int)[1,1,1] * 100

check_integral2 = Field(Tgrid_2_int)[1,1,1]/Field(Tgrid_2_ones_int)[1,1,1] * 100 < 1e-14

=#