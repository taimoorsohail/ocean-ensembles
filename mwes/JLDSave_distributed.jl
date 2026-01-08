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

arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=false)

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

checkpoint_intervals = IterationInterval(40)
##################### CHECKPOINTING (COMMENTED OUT) #####################
function save_restart(sim)
    @info @sprintf("Saving checkpoint file")
    @info sim.model.architecture
    if sim.model.architecture === nothing
        error("sim.model.architecture is not initialized; cannot determine local rank")
    end
    localrank = Integer(sim.model.architecture.local_rank)
    @info "Local rank: " * string(localrank)
    @info "Saving filename" * output_path * "checkpoint_sxtdeg_iteration" * string(sim.model.clock.iteration) * "_rank$(localrank).jld2"

    jldsave(output_path * "checkpoint_sxtdeg_iteration" * string(sim.model.clock.iteration) * "_rank$(localrank).jld2";
    u = on_architecture(CPU(), (sim.model.ocean.model.velocities.u)),
    v = on_architecture(CPU(), (sim.model.ocean.model.velocities.v)),
    w = on_architecture(CPU(), (sim.model.ocean.model.velocities.w)),
    T = on_architecture(CPU(), (sim.model.ocean.model.tracers.T)),
    S = on_architecture(CPU(), (sim.model.ocean.model.tracers.S)),
    e = on_architecture(CPU(), (sim.model.ocean.model.tracers.e)),
    η = on_architecture(CPU(), (sim.model.ocean.model.free_surface.η)),
    U = on_architecture(CPU(), (sim.model.ocean.model.free_surface.barotropic_velocities.U)),
    V = on_architecture(CPU(), (sim.model.ocean.model.free_surface.barotropic_velocities.V)),

    h = on_architecture(CPU(), (sim.model.sea_ice.model.ice_thickness)),
    ℵ = on_architecture(CPU(), (sim.model.sea_ice.model.ice_concentration)),
    σ₁₁ = on_architecture(CPU(), (sim.model.sea_ice.model.dynamics.auxiliaries.fields.σ₁₁)),
    σ₂₂ = on_architecture(CPU(), (sim.model.sea_ice.model.dynamics.auxiliaries.fields.σ₂₂)),
    σ₁₂ = on_architecture(CPU(), (sim.model.sea_ice.model.dynamics.auxiliaries.fields.σ₁₂)),
    Tu = on_architecture(CPU(), (sim.model.sea_ice.model.ice_thermodynamics.top_surface_temperature)),
    Gʰ = on_architecture(CPU(), (sim.model.sea_ice.model.ice_thermodynamics.thermodynamic_tendency)),
    u_ice = on_architecture(CPU(), (sim.model.sea_ice.model.velocities.u)),
    v_ice = on_architecture(CPU(), (sim.model.sea_ice.model.velocities.v)),
    clock = sim.model.ocean.model.clock)

    restartfiles = glob("jldsave_test_iteration*rank$(localrank)*", output_path)
    @info "restart files: " * string(restartfiles)
    # Extract the numeric suffix from each filename
    restart_numbers = map(f -> parse(Int, match(r"jldsave_test_iteration(\d+)", basename(f)).captures[1]), restartfiles)
    @info "Restart numbers: " * string(restart_numbers)
    sorted_restart_numbers = sort(unique(restart_numbers))

    # Keep only the last 50 iteration numbers
    if length(sorted_restart_numbers) < 50
        keep = sorted_restart_numbers
    else
        # Keep the last 50 iterations
        @info "Keeping last 50 restart files: " * string(sorted_restart_numbers[end-49:end])
        @info "Removing older restart files"
        keep = sorted_restart_numbers[end-49:end]
    end
    
    # Loop through and remove all older files for this rank
    for number in sorted_restart_numbers
        if number ∉ keep
            filename = output_path * "jldsave_test_iteration$(number)_rank$(localrank).jld2"
            if isfile(filename)
                @info "Removing old restart file: $filename"
                rm(filename; force = true)
            end
        end
    end
    @info "Done for rank $(localrank)"
end

add_callback!(simulation, save_restart, checkpoint_intervals)

##################### CHECKPOINTING (COMMENTED OUT) #####################

@info "Running simulation"
run!(simulation)
