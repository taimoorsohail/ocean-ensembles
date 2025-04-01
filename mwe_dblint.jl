using Oceananigans
using ClimaOcean

arch = CPU()

# ### Grid and Bathymetry
@info "Defining grid"

Nx = Integer(360)
Ny = Integer(180)
Nz = Integer(100)

@info "Defining vertical z faces"

r_faces = exponential_z_faces(; Nz, depth=5000, h=34)
z_faces = Oceananigans.MutableVerticalDiscretization(r_faces)

@info "Defining tripolar grid"

underlying_grid = TripolarGrid(arch;
                               size = (Nx, Ny, Nz),
                               z = z_faces,
                               halo = (5, 5, 4),
                               first_pole_longitude = 70,
                               north_poles_latitude = 55)

@info "Defining bottom bathymetry"

@time bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 75, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
				                  major_basins = 2)

# For this bathymetry at this horizontal resolution we need to manually open the Gibraltar strait.
view(bottom_height, 102:103, 124, 1) .= -400

@info "Defining grid"

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

c = CenterField(grid)
set!(c,1)

dblint = Field(Integral(Field(Integral(c, dims=1)), dims=(2,3))) # This works
compute!(dblint) # This doesn't
