using MPI
using CUDA
using CUDA: @allowscalar

MPI.Init()
atexit(MPI.Finalize)  

using ClimaOcean

using ClimaOcean.EN4
using ClimaOcean.ECCO
using ClimaOcean.EN4: download_dataset
using ClimaOcean.DataWrangling.ETOPO

using ClimaSeaIce
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Oceananigans.Operators: Ax, Ay, Az, Δz
using Oceananigans.Fields: ReducedField
using Oceananigans.Architectures: on_architecture

using OceanEnsembles

using CFTime
using Dates
using Printf
using Glob 
using JLD2
using CairoMakie

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

## Argument is provided by the submission script!

arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)

#total_ranks = MPI.Comm_size(MPI.COMM_WORLD)
localrank = Integer(arch.local_rank)
@info "Using architecture: " * string(arch)

# ### Grid and Bathymetry
@info "Defining grid"

Nx = Integer(360*6)
Ny = Integer(180*6)
Nz = Integer(75)

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


@info "Defining bottom bathymetry"

ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir = data_path)
ClimaOcean.DataWrangling.download_dataset(ETOPOmetadata)

bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
                                minimum_depth = 15,
                                interpolation_passes = 25, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
                                major_basins = 6)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)


fig = Figure(size = (800, 600))

ax1 = Axis(fig[1, 1], title = "Bathymetry", xlabel = "Longitude", ylabel = "Latitude")

hm = heatmap!(ax1, interior(grid.immersed_boundary.bottom_height)[:,:,1]; colormap = Reverse(:deep), colorrange = (-3, 0))
# lines!(ax1, [755, 1010, 1010, 755, 755], [800,790, 920,920, 800])
# lines!(ax1, [679, 670, 679, 688, 679], [878-6,881-6, 888-6,881-6, 878-6])
Colorbar(fig[1,2], hm; label = "Depth (m)")
save(figdir * "Bathymetry_masks.png", fig, px_per_unit=1)
