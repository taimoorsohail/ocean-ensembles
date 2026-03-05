using MPI
using CUDA
using NVTX

MPI.Init()
using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Printf
using ClimaOcean
using ClimaOcean.EN4
using ClimaOcean.ECCO
using Dates

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

Nx, Ny, Nz = Integer(360*6), Integer(180*6), 50

@info "Creating grid"
grid = TripolarGrid(arch; size = (Nx, Ny, Nz), z = (-6000, 0), halo = (7, 7, 4))
grid = analytical_immersed_grid(grid; active_cells_map=false)
free_surface = SplitExplicitFreeSurface(grid; substeps=50)

@info "Creating simulation"

ocean = ocean_simulation(grid; Δt=1minutes, free_surface, timestepper = :SplitRungeKutta3, closure = RiBasedVerticalDiffusivity())

dataset = EN4Monthly() # Other options include ECCO2Monthly(), ECCO4Monthly() or ECCO2Daily()

dates = vcat(collect(DateTime(1991, 1, 1): Month(1): DateTime(1991, 5, 1)),
             collect(DateTime(1990, 5, 1): Month(1): DateTime(1990, 12, 1)))

set!(ocean.model, T=Metadata(:temperature; dates=first(dates), dataset = dataset),
                  S=Metadata(:salinity;    dates=first(dates), dataset = dataset))

# Default sea-ice dynamics and salinity coupling are included in the defaults
sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7)) 

set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dataset=ECCO4Monthly()),
                    ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly()))

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(100), include_rivers_and_icebergs=true)
@info "Creating coupled model"

@time coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

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
for step in 1:250
    NVTX.@range "time step" begin
        time_step!(simulation)
    end
end