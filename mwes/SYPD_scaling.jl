using MPI
using CUDA

MPI.Init()
atexit(MPI.Finalize)  

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Printf
using Dates

arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=false)

function memory_status(arch::Union{Distributed{<:GPU}, <:GPU})
    free, total = CUDA.memory_info()
    used = total - free
    used_GiB, free_GiB, total_GiB = used / 2^30, free / 2^30, total / 2^30
    @show arch.local_rank, used_GiB, free_GiB, total_GiB
end

function memory_status(arch::Union{Distributed{<:CPU}, <:CPU})
    total = Sys.total_memory()
    free  = Sys.free_memory()
    used  = total - free
    used_GiB, free_GiB, total_GiB = used / 2^30, free / 2^30, total / 2^30
    @show arch.local_rank, used_GiB, free_GiB, total_GiB
end

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

function analytical_immersed_grid(underlying_grid::LatitudeLongitudeGrid;
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

Nx, Ny, Nz = 360,360,75
Lx, Ly = 100, 100

@info "Defining vertical z faces"

depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialDiscretization(Nz, depth, 0) 

memory_status(arch)

@info "Creating grid"

underlying_grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (6, 6, 3))


memory_status(arch)

@info "Defining grid"

grid = analytical_immersed_grid(underlying_grid; active_cells_map=true)

memory_status(arch)

@info "Creating free surface"
free_surface = SplitExplicitFreeSurface(grid; substeps = 70)

memory_status(arch)

@info "Creating model"

ocean_model = HydrostaticFreeSurfaceModel(; grid, free_surface, timestepper = :SplitRungeKutta3)

memory_status(arch)

@info "Creating simulation"

simulation = Simulation(ocean_model; Δt=10, verbose=false, stop_time=2hours)

memory_status(arch)

# Nice progress messaging is helpful:

## Print a progress message

wall_time = Ref(time_ns())

function progress(sim)
    step_time = 1e-9 * (time_ns() - wall_time[])
    wall_time[] = time_ns()
    @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
        iteration(sim), prettytime(sim), prettytime(sim.Δt),
        maximum(abs, sim.model.velocities.w), prettytime(step_time))
    return nothing
end

add_callback!(simulation, progress, TimeInterval(5minutes))

@info "Running simulation"

run!(simulation)

# @info "Simulation completed in " * prettytime(simulation.run_wall_time)