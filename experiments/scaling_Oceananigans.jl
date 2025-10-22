using MPI
using CUDA

MPI.Init()
atexit(MPI.Finalize)

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Printf
using JLD2
using SeawaterPolynomials.TEOS10

# include("load_balance_grid.jl")

if MPI.Comm_size(MPI.COMM_WORLD) == 1
    arch = GPU()
else
    arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal())) # , synchronized_communication=true)
end

function analytical_immersed_tripolar_grid(underlying_grid; radius = 5, active_cells_map = false) # degrees
    λp = 70 # underlying_grid.conformal_mapping.first_pole_longitude
    φp = 55 # underlying_grid.conformal_mapping.north_poles_latitude
    φm = -80 # underlying_grid.conformal_mapping.southernmost_latitude

    Lz = underlying_grid.Lz

    # We need a bottom height field that ``masks'' the singularities
    bottom_height(λ, φ) = ((abs(λ - λp) < radius)       & (abs(φp - φ) < radius)) |
                          ((abs(λ - λp - 180) < radius) & (abs(φp - φ) < radius)) | (φ < φm) ? 0 : - Lz

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map)

    return grid
end

Nx, Ny, Nz = 360*3, 180*3, 50

@info "Defining vertical z faces"

depth = -6000.0 # Depth of the ocean in meters

@info "Creating grid"

underlying_grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = (depth,0),
                    halo = (7, 7, 4))

# underlying_grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nz), z = (depth, 0), halo = (7, 7, 4), latitude = (-80, 80), longitude = (0, 360))

@info "Defining grid"

grid = analytical_immersed_tripolar_grid(underlying_grid; active_cells_map=true)

@info "Creating free surface"
free_surface = SplitExplicitFreeSurface(grid; substeps=50)

@info "Creating model"

model = HydrostaticFreeSurfaceModel(; grid,
                                      momentum_advection = WENOVectorInvariant(),
                                      tracer_advection = (T=WENO(order=7), S=WENO(order=7)), #, e=nothing),
                                      closure = RiBasedVerticalDiffusivity(),
                                      coriolis = HydrostaticSphericalCoriolis(),
                                      tracers = (:T, :S, :e),
                                      free_surface,
                                      buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState()))


set!(model, T=(x, y, z) -> rand()*1e-2 + 20, S=35)
ocean = Simulation(model, Δt = 0.1, stop_iteration = 1000)

wall_time = Ref(time_ns())

function progress_message(sim)
    η = sim.model.free_surface.η
    u, v, w = sim.model.velocities
    T, S = sim.model.tracers

    umax = (maximum(abs, (u)),
            maximum(abs, (v)),
            maximum(abs, (w)))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s,", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    msg7 = @sprintf("wall time: %s \n", prettytime(step_time))
    msg8 = @sprintf("SYPD: %.2f \n", (24*3600)/step_time/365)

    @info msg1 * msg2 * msg7 * msg8

    wall_time[] = time_ns()

    return nothing
end

add_callback!(ocean, progress_message, IterationInterval(100))

@info "Running simulation"

run!(ocean)