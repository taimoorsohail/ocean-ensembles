using MPI
using CUDA

MPI.Init()
atexit(MPI.Finalize)  

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Printf
using Dates
using Oceananigans.Architectures: on_architecture

using ClimaSeaIce
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium

using ClimaOcean
using ClimaOcean.ECCO
using JLD2

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")

arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=false)

function analytical_immersed_tripolar_grid(underlying_grid::TripolarGrid; radius = 5, active_cells_map = false) # degrees
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

Nx, Ny, Nz = 100, 100, 50
Lx, Ly = 100, 100

@info "Defining vertical z faces"

depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialDiscretization(Nz, depth, 0) 

@info "Creating grid"

underlying_grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (7, 7, 4))

@info "Defining grid"

grid = analytical_immersed_tripolar_grid(underlying_grid; active_cells_map=true)

@info "Creating free surface"
free_surface = SplitExplicitFreeSurface(grid; cfl=0.7, fixed_Δt=10minutes)

@info "Creating model"

@time ocean = ocean_simulation(grid; Δt=1minutes, free_surface, timestepper = :SplitRungeKutta3)

@info "Creating simulation"

sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7))

set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dataset=ECCO4Monthly(), dir=data_path),
                    ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly(), dir=data_path))

@info "Defining Atmospheric state"

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(100), include_rivers_and_icebergs=true)

@info "Defining coupled model"
@time coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

simulation = Simulation(coupled_model; Δt=10, verbose=false, stop_time=2hours)

progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.ocean.model.velocities.w), prettytime(sim.run_wall_time))

add_callback!(simulation, progress_message, IterationInterval(40))

@info "Running simulation"
run!(simulation)

##################### CHECKPOINTING (COMMENTED OUT) #####################
# function save_restart(sim)
#     localrank = MPI.Comm_rank(MPI.COMM_WORLD)
#     jldsave(output_path * "ocean_checkpointer_clock_iteration" * string(sim.model.clock.iteration) * "_rank$(localrank).jld2";
#     clock = sim.model.ocean.model.clock)
# end

# ocean_checkpointer_tracers = merge(
#     ocean.model.velocities,
#     ocean.model.tracers,
#     ocean.model.free_surface.barotropic_velocities,
#     (; η = simulation.model.ocean.model.free_surface.η)
# )
# sea_ice_checkpointer_tracers = merge(  
#                                 (ice_thickness = sea_ice.model.ice_thickness,
#                                 ice_concentration = sea_ice.model.ice_concentration,
#                                 top_surface_temperature = sea_ice.model.ice_thermodynamics.top_surface_temperature),
#                                 sea_ice.model.dynamics.auxiliaries.fields, 
#                                 sea_ice.model.velocities)

# iteration_number = string(simulation.model.clock.iteration)
# @time ocean.output_writers[:checkpointer] = JLD2Writer(ocean.model, ocean_checkpointer_tracers;
#                                             dir = output_path,
#                                             schedule = IterationInterval(40),
#                                             filename = "ocean_checkpointer_vars_iteration" * iteration_number,
#                                             overwrite_existing = true)

# @time sea_ice.output_writers[:checkpointer] = JLD2Writer(sea_ice.model, sea_ice_checkpointer_tracers;
#                                             dir = output_path,
#                                             schedule = IterationInterval(40),
#                                             filename = "sea_ice_checkpointer_vars_iteration" * iteration_number,
#                                             overwrite_existing = true)

##################### CHECKPOINTING (COMMENTED OUT) #####################

