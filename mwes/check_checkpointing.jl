using MPI
using CUDA
using CUDA: @allowscalar
MPI.Init()
atexit(MPI.Finalize) 

using NumericalEarth
using NumericalEarth.EN4
using NumericalEarth.ECCO
using NumericalEarth.EN4: download_dataset
using NumericalEarth.DataWrangling.ETOPO

using ClimaSeaIce
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Oceananigans.Operators: Ax, Ay, Az, Δz
using Oceananigans.Fields: ReducedField
using Oceananigans.Architectures: on_architecture

using Dates
using Printf
using JLD2

total_ranks = MPI.Comm_size(MPI.COMM_WORLD)

@show total_ranks

if total_ranks == 1
    arch = GPU()
else
    arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
end  

total_ranks = MPI.Comm_size(MPI.COMM_WORLD)
localrank = Integer(arch.local_rank)

# ### Grid and Bathymetry
@info "Defining grid"

Nx = Integer(40)
Ny = Integer(40)
ny = Ny/total_ranks
Nz = Integer(10)

@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialDiscretization(Nz, depth, 0, mutable=true) # IMPORTANT: WE NEED TO ACCOUNT FOR THIS

const z_surf = z_faces.cᵃᵃᶠ(Nz)
@info "Top grid cell is " * string(abs(round(z_surf))) * "m thick"

@info "Defining tripolar grid"
underlying_grid = TripolarGrid(arch;
                            size = (Nx, Ny, Nz),
                            z = z_faces,
                            halo = (7, 7, 7))


@time bottom_height = regrid_bathymetry(underlying_grid;
                                        minimum_depth = 15,
                                        interpolation_passes = 1, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
                                        major_basins = 4)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

@info "Defining ocean model"

free_surface = SplitExplicitFreeSurface(grid; substeps=10)

@time ocean = ocean_simulation(grid; Δt=1minutes,
                                momentum_advection = nothing,
                                tracer_advection = nothing,
                                free_surface)

set!(ocean.model, T=20, S=30)

#####
##### A Prognostic Sea-ice model
#####

# Default sea-ice dynamics and salinity coupling are included in the defaults
sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7))

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(10), include_rivers_and_icebergs=true)

# ### Coupled simulation

@time coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

simulation = Simulation(coupled_model; Δt=10minutes, stop_iteration = 5)

################################### START CHECKPOINTING ######################################

@time simulation.output_writers[:checkpointer] = Checkpointer(coupled_model,
                                                              schedule = IterationInterval(200), 
                                                              dir = "./",
                                                              prefix="RYF_sxtdeg_checkpoint_rank$localrank",
                                                              overwrite_existing = true,
                                                              cleanup = true)

################################### END CHECKPOINTING ######################################

@info "Running Simulation"

simulation.Δt = 10minutes
simulation.stop_time = 1days #parse(Int,ARGS[4]) * 13 * (365/12)days

run!(simulation, pickup=false, checkpoint_at_end=true)

### New simulation

@time ocean = ocean_simulation(grid; Δt=1minutes,
                                momentum_advection = nothing,
                                tracer_advection = nothing,
                                free_surface)

set!(ocean.model, T=20, S=30)

#####
##### A Prognostic Sea-ice model
#####

# Default sea-ice dynamics and salinity coupling are included in the defaults
sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7))

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(10), include_rivers_and_icebergs=true)

# ### Coupled simulation

@time coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

simulation = Simulation(coupled_model; Δt=10minutes, stop_iteration = 20)

@time simulation.output_writers[:checkpointer] = Checkpointer(coupled_model,
                                                              schedule = IterationInterval(200), 
                                                              dir = "./",
                                                              prefix="RYF_sxtdeg_checkpoint_rank$localrank",
                                                              overwrite_existing = true,
                                                              cleanup = true)

run!(simulation, pickup=true, checkpoint_at_end=true)
