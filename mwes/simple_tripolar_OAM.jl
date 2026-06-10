using CairoMakie
using NumericalEarth
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!

const Nx = 120
const Ny = 60
const Nz = 1

grid = TripolarGrid(CPU();
                    size = (Nx, Ny, Nz),
                    z = (-1, 0),
                    halo = (7, 7, 7))

atmosphere = PrescribedAtmosphere(grid, [0.0])
ocean = HydrostaticFreeSurfaceModel(grid; free_surface = SplitExplicitFreeSurface(grid, substeps=20))

u_field_atmos = atmosphere.velocities.u[1]
v_field_atmos = atmosphere.velocities.v[1]

u_field_ocean = ocean.velocities.u
v_field_ocean = ocean.velocities.v

set!(u_field_atmos, 1)
set!(v_field_atmos, 0)

set!(u_field_ocean, 1)
set!(v_field_ocean, 0)

fill_halo_regions!(u_field_ocean)
fill_halo_regions!(v_field_ocean)
fill_halo_regions!(u_field_atmos)
fill_halo_regions!(v_field_atmos)

fig = Figure(size = (1200, 900))

ax1 = Axis(fig[1, 1],
           title = "Prescribed atmosphere u at the tripolar seam",
           xlabel = "Seam index",
           ylabel = "Velocity")

lines!(ax1, u_field_atmos[:,Ny]; label = "Prescribed Atmosphere", linewidth = 3, color = :dodgerblue)
lines!(ax1, u_field_ocean[:,Ny,1]; label = "Ocean", linewidth = 3, color = :orangered)

outname = "simple_tripolar_OAM_seam.png"
axislegend(ax1, position = :rb)

save(outname, fig)

@info "Saved seam plot" outname
