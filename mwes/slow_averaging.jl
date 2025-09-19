using Oceananigans
using ClimaOcean

arch = CPU()
Nx = 360
Ny = 180
Nz = 40

depth = 4000
z = ExponentialCoordinate(Nz, -depth, 0; scale = 0.85*depth)

underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 1,
                                  major_basins = 2)

# We then incorporate the bathymetry into an ImmersedBoundaryGrid,

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                            active_cells_map=true)

field_immersed = Field{Center, Center, Nothing}(grid)
field_notimmersed = Field{Center, Center, Nothing}(underlying_grid)

# Loop over vars and times to compute averages
for i in 1:100
    set!(field_immersed, rand())
    set!(field_notimmersed, rand())
    avg_field_immersed = Average(field_immersed, dims = (1,2))
    avg_field_notimmersed = Average(field_notimmersed, dims = (1,2))

    @time test_immersed = Field(avg_field_immersed)[1,1,1]
    @time test_notimmersed = Field(avg_field_notimmersed)[1,1,1]
end
