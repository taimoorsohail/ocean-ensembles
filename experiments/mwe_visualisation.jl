using ClimaOcean
using Oceananigans
using Oceananigans.Units

using GLMakie

arch = CPU()

Nx = 360
Ny = 180
Nz = 30

underlying_grid = TripolarGrid(arch;
                               size = (Nx, Ny, Nz),
                               z = (-5000, 0),
                               halo = (5, 5, 4),
                               first_pole_longitude = 70,
                               north_poles_latitude = 55)

@time bottom_height = regrid_bathymetry(underlying_grid;
                                        minimum_depth = 10,
                                        interpolation_passes = 75, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
                                        major_basins = 2)

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

c = CenterField(grid)
set!(c, (λ, φ, z) -> (sind(3λ) + 1/3 * sind(5λ)) * cosd(3φ)^2)

land = interior(c.grid.immersed_boundary.bottom_height) .≥ 0
land = view(land, :, :, 1)

c_interior = interior(c, :, :, c.grid.Nz)
c_interior[land] .= NaN

cn = c_interior

fig = Figure(size=(700, 700))

axc = Axis(fig[1, 1])

cf = surface!(axc, λ, φ, ones(size(λ)), color=cn, colorrange=(-1, 1), colormap=:viridis, nan_color=:lightgray)

hidedecorations!(axc)
hidespines!(axc)

display(fig)
# save("mwe_sphere_tripolar.png", fig)