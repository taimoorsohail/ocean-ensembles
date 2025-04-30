using OceanEnsembles
using Oceananigans
using ClimaOcean
using Test

@testset "BasinMask" begin

    underlying_grid = TripolarGrid(size=(60, 40, 10),
                                   z = (-5000, 0),
                                   halo = (5, 5, 4),
                                   first_pole_longitude = 70,
                                   north_poles_latitude = 55)

    bottom_height = regrid_bathymetry(underlying_grid;
                                      minimum_depth = 10,
                                      interpolation_passes = 1,
                                      major_basins = 2)

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

    c = CenterField(grid)
    Atlantic_mask_ccc = basin_mask(grid, "atlantic", c)

    @test Atlantic_mask_ccc isa Array{Bool}
    @test size(view(c, :, :, 1)) == size(Atlantic_mask_ccc)

    w = ZFaceField(grid)
    Atlantic_mask_ccf = basin_mask(grid, "atlantic", w)

    @test Atlantic_mask_ccf isa Array{Bool}
    @test size(view(w, :, :, 1)) == size(Atlantic_mask_ccf)
end
