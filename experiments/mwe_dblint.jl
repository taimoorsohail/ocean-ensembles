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

@info "Defining grid"

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

@info "Defining free surface"

free_surface = SplitExplicitFreeSurface(grid; substeps=30)

momentum_advection = WENOVectorInvariant(vorticity_order=3)
tracer_advection   = Centered()

@info "Defining ocean simulation"

@time ocean = ocean_simulation(grid;
                         momentum_advection,
                         tracer_advection,
                         free_surface)


# c = CenterField(grid)
# volmask =  set!(c, 1)

# @info "Defining masks"

# Atlantic_mask = repeat(basin_mask(grid, "atlantic", c), 1, 1, Nz)
# IPac_mask = repeat(basin_mask(grid, "indo-pacific", c), 1, 1, Nz)
# glob_mask = Atlantic_mask .|| IPac_mask

# tot_int = compute!(Field(Integral(c, dims = (1,2,3), condition = glob_mask)))
# @info "The total volume is" tot_int[1,1,1]

# # using Oceananigans.Operators: Vccc
# # dvol_3D = [Vccc(i, j, k, grid) for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx]

# dvol_3D = yspacings(grid)*zspacings(grid)*xspacings(grid)
# basic_int = sum(c*dvol_3D*glob_mask)
# @info "The total volume is" basic_int

# basic_dbl_int = sum(sum(c*dvol_3D*glob_mask, dims=1), dims = (2,3))
# @info "The total volume is" basic_dbl_int[1, 1, 1]

# println(basic_int == basic_dbl_int[1, 1, 1])
# println(tot_int[1,1,1] == basic_dbl_int[1, 1, 1])
# println(tot_int[1,1,1] == basic_int)

# @info "Based on this test the best option is probably to multiply by a dV so that any reductions can be further reduced or divided by dV to get the average"