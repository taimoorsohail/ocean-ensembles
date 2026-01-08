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

arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)

total_ranks = MPI.Comm_size(MPI.COMM_WORLD)
localrank = Integer(arch.local_rank)
@info "Using architecture: " * string(arch)

# ### Grid and Bathymetry
@info "Defining grid"

Nx = Integer(360*6)
Ny = Integer(180*6)
ny = Integer(Ny/total_ranks)
@show ny
@show total_ranks
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

@info "Applying bathymetry masks"
# Black + Caspian Seas
xs1 = [755, 1010, 1010, 755]
# We split the y-polygon because it stretches across two ranks [unique to 4 rank run :(]
ys1_1 = [800,  790,  809,  809]
ys1_2 = [810,  810,  920,  920]

# Baltic Sea / Danish Straits
xs2 = [679, 670-9, 679, 688+9]
ys2 = [872-10, 875+6, 882+5, 875+6]

nys1_1 = ys1_1/ny
ys1_1_floor = floor.(nys1_1)

nys1_2 = ys1_2/ny
ys1_2_floor = floor.(nys1_2)

nys2 = ys2/ny
ys2_floor = floor.(nys2)

length(unique(ys1_1_floor)) == 1 || error("Mask must all lie in the same rank! Currently they lie in ranks $(unique(ys1_1_floor))")

length(unique(ys1_2_floor)) == 1 || error("Mask must all lie in the same rank! Currently they lie in ranks $(unique(ys1_2_floor))")

length(unique(ys2_floor)) == 1 || error("Mask must all lie in the same rank! Currently they lie in ranks $(unique(ys2_floor))")


if localrank == Integer(unique(ys1_1_floor)[1])
    bh = interior(bottom_height)[:,:,1]
    @show Integer(unique(ys1_1_floor)[1]), localrank, Int.(ys1_1 .- ny * localrank), Ny, ny
    mask_blacksea_caspian_1 = section_mask(xs1, Int.(ys1_1 .- ny * localrank), ones(length(xs1)), underlying_grid)
    idx = (CuArray(mask_blacksea_caspian_1) .== 1) .& (bh .<= 0)
    bh[idx] .= 0;
    interior(bottom_height) .= bh
    # set!(bottom_height, bh)
    fig = Figure(size = (360*6*5, 1600))
    ax1 = Axis(fig[1, 1], title = "Bathymetry", xlabel = "Longitude", ylabel = "Latitude")
    hm = heatmap!(ax1, Array(interior(bottom_height)[:,:,1]); colormap = Reverse(:seismic), colorrange = (-3f0,0f0))
    # Colorbar(fig[1,2], hm; label = "Depth (m)")
    @allowscalar save(figdir * "Caspian_bathy_rank$(localrank).png", fig, px_per_unit=1)
end 

if localrank == Integer(unique(ys1_2_floor)[1])
    bh = interior(bottom_height)[:,:,1]

    @show Integer(unique(ys1_2_floor)[1]), localrank, Int.(ys1_2 .- ny * localrank), Ny, ny, xs1

    mask_blacksea_caspian_2 = section_mask(xs1, Int.(ys1_2 .- ny * localrank), ones(length(xs1)), underlying_grid)
    
    idx = (CuArray(mask_blacksea_caspian_2) .== 1) .& (bh .<= 0)
    bh[idx] .= 0;
    interior(bottom_height) .= bh
    fig = Figure(size = (360*6*5, 1600))
    ax1 = Axis(fig[1, 1], title = "Bathymetry", xlabel = "Longitude", ylabel = "Latitude")
    hm = heatmap!(ax1, Array(mask_blacksea_caspian_2); colormap = Reverse(:seismic), colorrange = (0f0,1f0))
    # Colorbar(fig[1,2], hm; label = "Depth (m)")
    @allowscalar save(figdir * "Caspian_mask_rank$(localrank).png", fig, px_per_unit=1)
    fig = Figure(size = (360*6*5, 1600))
    ax1 = Axis(fig[1, 1], title = "Bathymetry", xlabel = "Longitude", ylabel = "Latitude")
    hm = heatmap!(ax1, Array(interior(bottom_height)[:,:,1]); colormap = Reverse(:seismic), colorrange = (-3f0,0f0))
    # Colorbar(fig[1,2], hm; label = "Depth (m)")
    @allowscalar save(figdir * "Caspian_bathy_rank$(localrank).png", fig, px_per_unit=1)

end
if localrank == Integer(unique(ys2_floor)[1])
    bh = interior(bottom_height)[:,:,1]

    @show Integer(unique(ys2_floor)[1]), localrank, Int.(ys2 .- ny * localrank)
    mask_danish_strait = section_mask(xs2, Int.(ys2 .- ny * localrank), ones(length(xs2)).*2, underlying_grid)
    idx = (CuArray(mask_danish_strait) .== 2) .& (bh .>= 0) .& (bh .<= 3)
    bh[idx] .= -10;
    interior(bottom_height) .= bh
end

@info "Creating Grid"

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

fig = Figure(size = (360*6*5, 1600))

ax1 = Axis(fig[1, 1], title = "Bathymetry", xlabel = "Longitude", ylabel = "Latitude")
# ax2 = Axis(fig[1, 2], title = "Bathymetry", xlabel = "Longitude", ylabel = "Latitude")
# ax3 = Axis(fig[1, 3], title = "Bathymetry", xlabel = "Longitude", ylabel = "Latitude")

hm = heatmap!(ax1, Array(interior(grid.immersed_boundary.bottom_height)[:,:,1]); colormap = Reverse(:seismic), colorrange = (-3,0))

# bh_matrix[(mask_blacksea_caspian .== 1) .& (bh_matrix .<= 0)] .= 0;
# hm = heatmap!(ax2, bh_matrix; colormap = Reverse(:seismic), colorrange = (-3,0))

# bh_matrix[(mask_danish_strait .== 2) .& (bh_matrix .>= 0) .& (bh_matrix .<= 3)] .= -10;
# @show minimum(bh_matrix)
# hm = heatmap!(ax3, bh_matrix; colormap = Reverse(:seismic), colorrange = (-3, 0))

# Colorbar(fig[1,2], hm; label = "Depth (m)")
@allowscalar save(figdir * "Bathymetry_masks_distributed_rank$(localrank).png", fig, px_per_unit=1)