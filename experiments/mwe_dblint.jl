using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
using OceanEnsembles

Nx = Integer(360/4)
Ny = Integer(180/4)
Nz = Integer(100/4)

arch = CPU()

z_faces = (-4000, 0)

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

@info "Defining free surface"

c = CenterField(grid)
volmask =  set!(c, 1)

@info "Defining masks"

Atlantic_mask = repeat(basin_mask(grid, "atlantic", c), 1, 1, Nz)
IPac_mask = repeat(basin_mask(grid, "indo-pacific", c), 1, 1, Nz)
glob_mask = Atlantic_mask .|| IPac_mask

tot_int = compute!(Field(Integral(c, dims = (1,2,3), condition = glob_mask)))
@info "The total integral of 1s is", tot_int

# sing_int = Field(Integral(c, dims = (1), condition = glob_mask))
# dArea = grid. 
# tot_int_from_sing = sum(sing_int*dArea)