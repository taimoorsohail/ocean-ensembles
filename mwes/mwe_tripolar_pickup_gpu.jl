using CUDA
using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: interior

const OUTPUT_DIR = mkpath(joinpath(tempdir(), "oceananigans_pickup_mwe"))
const PREFIX = "tripolar_pickup_mwe"

function analytical_immersed_grid(arch; size = (96, 48, 8), halo = (7, 7, 7))
    underlying_grid = TripolarGrid(arch;
                                   size,
                                   z = (-100, 0),
                                   halo,
                                   fold_topology = RightFaceFolded)

    function bottom_height(λ, φ)
        λp = underlying_grid.conformal_mapping.first_pole_longitude
        φp = underlying_grid.conformal_mapping.north_poles_latitude
        φm = underlying_grid.conformal_mapping.southernmost_latitude
        Lz = underlying_grid.Lz
        singular = ((abs(λ - λp) < 5) && (abs(φp - φ) < 5)) ||
                   ((abs(λ - λp - 180) < 5) && (abs(φp - φ) < 5)) ||
                   (φ < φm)
        return singular ? 0 : -Lz
    end

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map = true)
end

function build_simulation(substeps, output_dir)
    arch = GPU()
    grid = analytical_immersed_grid(arch)
    free_surface = SplitExplicitFreeSurface(grid; substeps)
    model = HydrostaticFreeSurfaceModel(grid;
                                        free_surface,
                                        timestepper = :SplitRungeKutta3)

    simulation = Simulation(model; Δt = 10minutes, stop_iteration = 1)
    simulation.output_writers[:checkpointer] = Checkpointer(model;
                                                            schedule = IterationInterval(1),
                                                            dir = output_dir,
                                                            prefix = PREFIX,
                                                            overwrite_existing = true,
                                                            cleanup = false)
    return simulation
end

function finite_state(simulation)
    u = Array(interior(simulation.model.velocities.u))
    v = Array(interior(simulation.model.velocities.v))
    η = Array(interior(simulation.model.free_surface.displacement))
    return all(isfinite, u) && all(isfinite, v) && all(isfinite, η)
end

function run_phase(substeps, pickup, stop_iteration, output_dir)
    simulation = build_simulation(substeps, output_dir)
    simulation.stop_iteration = stop_iteration
    run!(simulation; pickup, checkpoint_at_end = true)
    ok = finite_state(simulation)
    iter = simulation.model.clock.iteration
    simulation = nothing
    GC.gc(true)
    CUDA.reclaim()
    return iter, ok
end

rm.(filter(contains(PREFIX), readdir(OUTPUT_DIR; join = true)); force = true)

iter1, ok1 = run_phase(12, false, 1, OUTPUT_DIR)
iter2, ok2 = run_phase(20, true, 2, OUTPUT_DIR)

println("MWE_PHASE1 iteration=$(iter1) finite=$(ok1)")
println("MWE_PHASE2 iteration=$(iter2) finite=$(ok2)")

if iter2 >= 2 && ok1 && ok2
    println("MWE_PICKUP_SUCCESS final_iteration=$(iter2) output_dir=$(OUTPUT_DIR)")
else
    error("MWE pickup failed: iteration=$(iter2) finite1=$(ok1) finite2=$(ok2)")
end
