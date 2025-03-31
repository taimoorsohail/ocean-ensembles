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

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

c = CenterField(grid)
set!(c, (λ, φ, z) -> (sind(3λ) + 1/3 * sind(5λ)) * cosd(3φ)^2)

import Oceananigans.Grids: nodes

function nodes(grid::OrthogonalSphericalShellGrid, ℓx, ℓy, ℓz; reshape=false, with_halos=false)
    λ = λnodes(grid, ℓx, ℓy, ℓz; with_halos)
    φ = φnodes(grid, ℓx, ℓy, ℓz; with_halos)
    z = znodes(grid, ℓx, ℓy, ℓz; with_halos)

    if reshape
        # λ and φ are 2D arrays
        N = (size(λ)..., size(z)...)
        λ = Base.reshape(λ, N[1], N[2], 1)
        φ = Base.reshape(φ, N[1], N[2], 1)
        z = Base.reshape(z, 1, 1, N[3])
    end

    return (λ, φ, z)
end


function geographic2cartesian(c::Field; r=1)

    λ, φ, z = nodes(c)

    x = @. r * cosd(λ) * cosd(φ)
    y = @. r * sind(λ) * cosd(φ)
    z = @. r * sind(φ)

    return x, y, z
end

x, y, z = geographic2cartesian(c)

land = interior(c.grid.immersed_boundary.bottom_height) .≥ 0
land = view(land, :, :, 1)

c_interior = interior(c, :, :, c.grid.Nz)
c_interior[land] .= NaN

cn = c_interior

fig = Figure(size=(700, 700))
kw = (elevation= deg2rad(-10), azimuth=deg2rad(90), aspect=:equal)

axc = Axis3(fig[1, 1]; kw...)

cf = surface!(axc, x, y, z, color=cn, colorrange=(-1, 1), colormap=:viridis, nan_color=:lightgray)
Colorbar(fig[2, 1], cf, width=Relative(0.6), vertical=false, label = "tracer", labelsize=20)

hidedecorations!(axc)
hidespines!(axc)

display(fig)
# save("mwe_sphere_tripolar.png", fig)