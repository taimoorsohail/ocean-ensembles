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
using Printf: @sprintf

const Nx = Integer(360)
const Ny = Integer(180)
const Nz = round(Int, 75)
const depth = -5500.0
const output_path = expanduser("/home/tsohail/uom/ocean-ensembles/outputs/")
const checkpoint_prefix = "simple_tripolar_OOM"

function progress(sim)
    clock = sim.model.clock
    msg = @sprintf("progress | iter=%d stage=%d time=%.6f last_Δt=%.6f wall=%.2fs",
                   clock.iteration,
                   clock.stage,
                   Float64(clock.time),
                   Float64(clock.last_Δt),
                   sim.run_wall_time)
    println(msg)
    flush(stdout)
    return nothing
end

@info "Defining underlying grid"
underlying_grid = TripolarGrid(CPU();
                               size = (Nx, Ny, Nz),
                               z = (-100, 0),
                               halo = (7, 7, 7), 
                               fold_topology = RightCenterFolded)

@info "Defining immersed grid"
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
@info "Applying analytical immersed grid"
grid = analytical_immersed_grid(underlying_grid; radius = 5, active_cells_map = true) # degrees
@info "Defining free surface and advection schemes"
fs = SplitExplicitFreeSurface(grid, substeps=100)
momentum_advection = WENOVectorInvariant(time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5))
tracer_advection = WENO(order=7, time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl=0.5))
@info "Defining ocean model and simulation"
@time ocean = ocean_simulation(grid; free_surface=fs, 
                                   momentum_advection,
                                   tracer_advection,
                                   timestepper=:SplitRungeKutta3,
)

model = OceanOnlyModel(ocean)
simulation = Simulation(model, Δt=10minutes)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))
@info "Adding checkpointer to output writers"
simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule=IterationInterval(50),
                                                        dir=output_path,
                                                        prefix=checkpoint_prefix,
                                                        overwrite_existing=true,
                                                        cleanup=false)
@info "Running simulation"
simulation.stop_iteration = 10
run!(simulation, pickup=true)

@info "Simulation complete"