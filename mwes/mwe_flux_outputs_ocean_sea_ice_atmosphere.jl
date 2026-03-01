using Oceananigans
using Oceananigans.OutputWriters: JLD2Writer
using Oceananigans.Units: minutes
using JLD2

using NumericalEarth
using NumericalEarth.Atmospheres: PrescribedAtmosphere

const OUTPUT_DIR = "/g/data/v46/txs156/ocean-ensembles/outputs"
const OUTPUT_FILE = "mwe_flux_outputs_5x5x2"

arch = CPU()

Nx, Ny, Nz = 5, 5, 2
grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             longitude = (0, 5),
                             latitude = (-75, -70),
                             z = (-20, 0),
                             topology = (Bounded, Bounded, Bounded))

ocean = ocean_simulation(grid;
                         momentum_advection = nothing,
                         tracer_advection = nothing,
                         coriolis = nothing,
                         closure = nothing,
                         bottom_drag_coefficient = 0,
                         radiative_forcing = nothing)

set!(ocean.model, u = 0, v = 0, T = 1, S = 34.5)

sea_ice = sea_ice_simulation(grid, ocean)
set!(sea_ice.model, h = 0.5, ℵ = 0.8)

times = [0.0, 1.0minutes]
atmosphere = PrescribedAtmosphere(grid, times)

# Constant prescribed atmospheric state for a minimal reproducible run.
fill!(parent(atmosphere.tracers.T), 268.15)
fill!(parent(atmosphere.tracers.q), 0.004)
fill!(parent(atmosphere.velocities.u), 3.0)
fill!(parent(atmosphere.velocities.v), 1.0)
fill!(parent(atmosphere.pressure), 101325.0)
fill!(parent(atmosphere.downwelling_radiation.shortwave), 250.0)
fill!(parent(atmosphere.downwelling_radiation.longwave), 300.0)
fill!(parent(atmosphere.freshwater_flux.rain), 0.0)
fill!(parent(atmosphere.freshwater_flux.snow), 0.0)

coupled_model = OceanSeaIceModel(ocean, sea_ice;
                                 atmosphere,
                                 radiation = Radiation(arch))

ao_fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

flux_outputs = (; total_heat_flux = ao_fluxes.total_heat_flux,
                 total_freshwater_flux = ao_fluxes.total_freshwater_flux,
                 total_freshwater_flux_with_salt_equiv = ao_fluxes.total_freshwater_flux_with_salt_equiv)

surface_height = (; surface_height = ocean.model.free_surface.displacement)
surface_forcing = (; T_surf = ocean.model.tracers.T.boundary_conditions.top.condition,
                    S_surf = ocean.model.tracers.S.boundary_conditions.top.condition,
                    total_heat_flux = ao_fluxes.total_heat_flux,
                    total_freshwater_flux = ao_fluxes.total_freshwater_flux,
                    total_freshwater_flux_with_salt_equiv = ao_fluxes.total_freshwater_flux_with_salt_equiv)
outputs_surf = merge(surface_height, surface_forcing)

simulation = Simulation(coupled_model;
                        Δt = 10.0,
                        stop_iteration = 2,
                        verbose = false)

simulation.output_writers[:fluxes] = JLD2Writer(coupled_model, flux_outputs;
                                                dir = OUTPUT_DIR,
                                                filename = OUTPUT_FILE,
                                                schedule = IterationInterval(1),
                                                overwrite_existing = true,
                                                with_halos = false)

ocean.output_writers[:surface_forcing] = JLD2Writer(ocean.model, outputs_surf;
                                                    dir = OUTPUT_DIR,
                                                    filename = OUTPUT_FILE * "_surface_forcing",
                                                    schedule = IterationInterval(1),
                                                    overwrite_existing = true,
                                                    with_halos = false)

run!(simulation)

filepath = joinpath(OUTPUT_DIR, OUTPUT_FILE * ".jld2")
surface_filepath = joinpath(OUTPUT_DIR, OUTPUT_FILE * "_surface_forcing.jld2")

jldopen(filepath, "r") do file
    println("Saved output file: ", filepath)
    println("Top-level keys: ", collect(keys(file)))
end

jldopen(surface_filepath, "r") do file
    println("Saved surface-forcing file: ", surface_filepath)
    println("Surface-forcing timeseries keys: ", collect(keys(file["timeseries"])))
end

println("Completed 2 iterations. Flux diagnostics were written with JLD2Writer.")
