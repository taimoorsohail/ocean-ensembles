using CairoMakie
using NumericalEarth
using NumericalEarth.EN4
using NumericalEarth.ECCO
using NumericalEarth.DataWrangling.ETOPO
using ClimaSeaIce
using CUDA
using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: interior

const Nx = Integer(360)
const Ny = Integer(180)
const Nz = round(Int, 75)
const depth = -5500.0

underlying_grid = TripolarGrid(CPU();
                               size = (Nx, Ny, Nz),
                               z = (-100, 0),
                               halo = (7, 7, 7), 
                               fold_topology = RightCenterFolded)

function analytical_immersed_grid(underlying_grid::TripolarGrid; radius = 5, active_cells_map = false) # degrees
    λp = underlying_grid.conformal_mapping.first_pole_longitude
    φp = underlying_grid.conformal_mapping.north_poles_latitude
    φm = underlying_grid.conformal_mapping.southernmost_latitude

    Lz = underlying_grid.Lz

    # We need a bottom height field that ``masks'' the singularities
    bottom_height(λ, φ) = ((abs(λ - λp) < radius)       & (abs(φp - φ) < radius)) |
                          ((abs(λ - λp - 180) < radius) & (abs(φp - φ) < radius)) | (φ < φm) ? 0 : - Lz

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map)

    return grid
end

grid = analytical_immersed_grid(underlying_grid; radius = 5, active_cells_map = true) # degrees

fs = SplitExplicitFreeSurface(grid, substeps=70)

@time ocean = ocean_simulation(grid; free_surface=fs)

model = OceanOnlyModel(ocean)
simulation = Simulation(model, Δt=10minutes)

simulation.stop_time = 1days
run!(simulation)

lines(interior(simulation.model.velocities.w, :, Ny, 75))