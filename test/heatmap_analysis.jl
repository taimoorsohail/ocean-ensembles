using ClimaOcean
using Oceananigans

arch = CPU()#Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
Nx, Ny, Nz = 30,30,30#360*6, 180*6, 75
@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
r_faces = ExponentialCoordinate(Nz, depth, 0)
@info "Defining grid"

# grid = TripolarGrid(arch;
#                     size = (Nx, Ny, 1),
#                     # z = r_faces,
#                     halo = (6, 6, 3),
#                     first_pole_longitude = 70,
#                     north_poles_latitude = 55)

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, 1),
                             halo = (7, 7, 7),
                             z = (0,1),
                             latitude  = (-75, 75),
                             longitude = (0, 360))

Tgrid = CenterField(grid)

function temp_lon_spherical(λ, φ, z)
    #This function should integrate to zero
    return sin(λ) + cos(φ)
end

set!(Tgrid, temp_lon_spherical)
Tgrid_int = compute!(Integral(Tgrid))
@show Field(Tgrid_int)[1,1,1]