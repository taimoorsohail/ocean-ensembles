using Oceananigans
using CairoMakie

Nx, Ny, Nz = 60, 30, 20

grid = Oceananigans.OrthogonalSphericalShellGrids.TripolarGrid(CPU(); size=(Nx, Ny, Nz),
                                                                      halo=(7, 7, 7),
                                                                      z=(-1000, 0))


c = CenterField(grid)

set!(c, 1)

∫c = Field(Average(c, dims=1))
compute!(∫c)