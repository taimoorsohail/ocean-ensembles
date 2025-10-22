using MPI
using CUDA
using NVTX

MPI.Init()
using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Printf
using ClimaOcean

if MPI.Comm_size(MPI.COMM_WORLD) == 1
    arch = GPU()
else
    arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=false)
end

function analytical_immersed_grid(underlying_grid; radius = 5, active_cells_map = false) # degrees
    λp = 70
    φp = 55
    φm = -80
    Lz = underlying_grid.Lz
    bottom_height(λ, φ) = ((abs(λ - λp) < radius)       & (abs(φp - φ) < radius)) |
                          ((abs(λ - λp - 180) < radius) & (abs(φp - φ) < radius)) | (φ < φm) ? 0 : - Lz
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map)
    return grid
end

Nx, Ny, Nz = 360*3, 180*3, 50

@info "Creating grid"
grid = TripolarGrid(arch; size = (Nx, Ny, Nz), z = (-6000, 0), halo = (7, 7, 4))
grid = analytical_immersed_grid(grid; active_cells_map=false)
free_surface = SplitExplicitFreeSurface(grid; substeps=50)
ocean = ocean_simulation(grid; Δt=1minutes, free_surface, timestepper = :SplitRungeKutta3, closure = RiBasedVerticalDiffusivity())
set!(ocean.model, T=(x, y, z) -> rand()*1e-2 + 20, S=35)

@info "Creating simulation"

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(100), include_rivers_and_icebergs=true)

@time coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=10, verbose=false, stop_iteration=100)

@info "Running simulation"

# Precompilation steps
for step in 1:20
   time_step!(simulation)
end

# Make sure all cores are aligned
GC.gc()
MPI.Barrier(MPI.COMM_WORLD)

# Profile!
for step in 1:500
    NVTX.@range "time step" begin
        time_step!(simulation)
        MPI.Barrier(MPI.COMM_WORLD)
        if step % 50 == 0
            GC.gc()
        end
    end
end