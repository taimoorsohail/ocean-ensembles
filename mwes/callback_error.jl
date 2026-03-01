using MPI
using CUDA

MPI.Init()
atexit(MPI.Finalize)  

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Printf
using Dates
using NumericalEarth

arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=false)

function immersed_latlon_grid(underlying_grid::LatitudeLongitudeGrid;
                                           radius = 5,  # controls width of Gaussian (in degrees)
                                           height = 2000, # hill height in meters
                                           λc = 0, φc = 45, # hill center (lon, lat)
                                           active_cells_map = false)

    Lz = underlying_grid.Lz

    # Convert degrees to radians if grid expects radians, but leaving in degrees here
    bottom_height(λ, φ) = begin
        # Gaussian hill centered at (λc, φc)
        r² = (λ - λc)^2 + (φ - φc)^2
        z = -Lz + height * exp(-r² / (2 * radius^2))
        return z
    end

    grid = ImmersedBoundaryGrid(underlying_grid,
                                GridFittedBottom(bottom_height);
                                active_cells_map)

    return grid
end

Nx, Ny, Nz = 10, 10, 10
Lx, Ly = 100, 100

@info "Defining vertical z faces"

depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialDiscretization(Nz, depth, 0) 

@info "Creating grid"

underlying_grid = LatitudeLongitudeGrid(arch,
                                        size = (Nx, Ny, Nz),
                                        z = z_faces,
                                        halo = (7, 7, 4),
                                        longitude = (0, 360),
                                        latitude = (-70, 70))



@info "Defining grid"

grid = immersed_latlon_grid(underlying_grid; active_cells_map=true)
@show grid

@info "Creating free surface"
free_surface = SplitExplicitFreeSurface(grid; substeps = 70)

@info "Creating model"

ocean = ocean_simulation(grid; Δt=1minutes, timestepper = :SplitRungeKutta3, free_surface)

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(100), include_rivers_and_icebergs=true)
sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7)) 

@info "Creating simulation"
@time coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=10minutes)

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = IterationInterval(5)

function progress(sim)
    η = sim.model.ocean.model.free_surface.displacement
    u, v, w = sim.model.ocean.model.velocities
    T, S = sim.model.ocean.model.tracers

    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    ηrange = (maximum((η)), minimum((η)))

    umax = (maximum(abs, (u)),
            maximum(abs, (v)),
            maximum(abs, (w)))

    step_time = 1e-9 * (time_ns() - wall_time[])
    wall_progress = time_ns() * 1e-9

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s,", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Trange...)
    msg4 = @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Srange...)
    msg6 = @sprintf("extrema(η): (%.2f, %.2f) m, ", ηrange...)
    msg7 = @sprintf("wall time: %s \n", prettytime(step_time))
    msg5 = @sprintf("Wall clock time: %s \n", prettytime(wall_progress))
    msg8 = @sprintf("SYPD: %.2f \n", (10*sim.Δt)/step_time/365)

    @info msg1 * msg2 * msg3 * msg4 * msg6 * msg7 * msg5 * msg8

    wall_time[] = time_ns()

    return nothing
end

add_callback!(coupled_model, progress, callback_interval)

@info "Running simulation"
simulation.stop_iteration = 20
run!(simulation)

# @info "Simulation completed in " * prettytime(simulation.run_wall_time)