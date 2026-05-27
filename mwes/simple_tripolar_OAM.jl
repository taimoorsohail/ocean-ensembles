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


time_indices_in_memory = 24
arch = GPU()

underlying_grid = TripolarGrid(arch;
                               size = (Nx, Ny, Nz),
                               z = (-100, 0),
                               halo = (7, 7, 7))

grid = analytical_immersed_grid(underlying_grid; radius = 5, active_cells_map = true) # degrees

@time ocean = ocean_simulation(grid; free_surface=SplitExplicitFreeSurface(grid, substeps=70))

radiation = JRA55PrescribedRadiation(arch; time_indices_in_memory)
atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory)

model = OceanOnlyModel(ocean; atmosphere, radiation)
simulation = Simulation(model, Δt=10minutes)

surface_forcing = (; heat_flux=(net_ocean_heat_flux(simulation.model)),
                    fw_flux=(net_ocean_freshwater_flux(simulation.model)))

@time simulation.output_writers[:surface_fluxes] = JLD2Writer(simulation.model, surface_forcing;
                                                                schedule=IterationInterval(1),
                                                                filename="test_surface_luxes_plotting",
                                                                with_halos=false,
                                                                overwrite_existing=true,
                                                                array_type=Array{Float32})

simulation.stop_iteration = 5
run!(simulation)