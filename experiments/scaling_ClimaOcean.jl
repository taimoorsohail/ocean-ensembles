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

println("MPI CUDA aware: ", MPI.has_cuda())

if MPI.Comm_size(MPI.COMM_WORLD) == 1
    arch = CPU()
else
    arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal(), x = DistributedComputations.Equal()), synchronized_communication=true)
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

Nx, Ny, Nz = 360*6, 180*6, 50

@info "Creating grid"
# grid =  LatitudeLongitudeGrid(arch,
#                               size = (Nx, Ny, Nz),
#                               z = (depth,0),
#                               halo = (7, 7, 4),
#                               longitude = (0, 360),
#                               latitude = (-70, 70))
                                    
grid = TripolarGrid(arch; size = (Nx, Ny, Nz), z = (-6000, 0), halo = (7, 7, 4))
grid = analytical_immersed_grid(grid; active_cells_map=false)
free_surface = SplitExplicitFreeSurface(grid; substeps=50)
ocean = ocean_simulation(grid; Δt=1minutes, free_surface, timestepper = :SplitRungeKutta3, closure = RiBasedVerticalDiffusivity())

dataset = EN4Monthly() # Other options include ECCO2Monthly(), ECCO4Monthly() or ECCO2Daily()
dates = vcat(collect(DateTime(1991, 1, 1): Month(1): DateTime(1991, 5, 1)),
             collect(DateTime(1990, 5, 1): Month(1): DateTime(1990, 12, 1)))

set!(ocean.model, T=Metadata(:temperature; dates=first(dates), dataset = dataset),
                  S=Metadata(:salinity;    dates=first(dates), dataset = dataset))

@info "Creating simulation"

# Default sea-ice dynamics and salinity coupling are included in the defaults
sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7)) 

set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dataset=ECCO4Monthly()),
                    ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly()))

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
for step in 1:200
    NVTX.@range "time step" begin
        time_step!(simulation)
    end
end