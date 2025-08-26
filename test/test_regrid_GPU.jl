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

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")

## Argument is provided by the submission script!

arch = GPU()

#total_ranks = MPI.Comm_size(MPI.COMM_WORLD)

@info "Using architecture: " * string(arch)

# ### Download necessary files to run the code

# ### ECCO files
@info "Downloading/checking input data"

dates = vcat(collect(DateTime(1991, 1, 1): Month(1): DateTime(1991, 5, 1)),
             collect(DateTime(1990, 5, 1): Month(1): DateTime(1990, 12, 1)))

@info "We download the 1990-1991 data for an RYF implementation"

dataset = EN4Monthly() # Other options include ECCO2Monthly(), ECCO4Monthly() or ECCO2Daily()

temperature = Metadata(:temperature; dates, dataset = dataset, dir=data_path)
salinity    = Metadata(:salinity;    dates, dataset = dataset, dir=data_path)

download_dataset(temperature)
download_dataset(salinity)

# ### Grid and Bathymetry
@info "Defining grid"

Nx = Integer(360)
Ny = Integer(180)
Nz = Integer(75)

@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialCoordinate(Nz, depth, 0)

const z_surf = z_faces(Nz)


@info "Top grid cell is " * string(abs(round(z_faces(Nz)))) * "m thick"

# z_faces = Oceananigans.Grids.MutableVerticalDiscretization(z_faces)

@info "Defining tripolar grid"

grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (7, 7, 7))

@info "Defining destination underlying grid"
destination_grid = LatitudeLongitudeGrid(arch;
                                        size = (Nx, Ny, Nz),
                                        halo = (7, 7, 7),
                                        z = z_faces,
                                        latitude  = (-80, 90),
                                        longitude = (0, 360))


destination_field = Field{Center, Center, Center}(destination_grid)
source_field = Field{Center, Center, Center}(grid)

f(x,y,z) = sin(y)^2+cos(x)^2 # Example function to set the source field
set!(source_field, f)

@info "Defining regridder weights"
@time W = @allowscalar regridder_weights(source_field, destination_field; method = "conservative")

src_flat = vec(permutedims(collect(interior(source_field))[:,:,1]))  # shape (ncell, 1)

# Regrid the source field to the destination grid
dst_vec = reshape((W) * src_flat, destination_field.grid.Nx, destination_field.grid.Ny)

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

using CairoMakie
# Plotting the regridded data vs the source data
fig1 = Figure(size = (1600, 800))
ax1 = Axis(fig1[1, 1], title="Source Field", xlabel="i", ylabel="j")
ax2 = Axis(fig1[1, 3], title="Regridding Weights - Conservative", xlabel="i", ylabel="j")
ax3 = Axis(fig1[1, 5], title="Destination Field - Conservative", xlabel="Longitude", ylabel="Latitude")
# ax4 = Axis(fig1[2, 1], title="Regridding Weights - Bilinear", xlabel="Longitude", ylabel="Latitude")
# ax5 = Axis(fig1[2, 3], title="Destination Field - Bilinear", xlabel="Longitude", ylabel="Latitude")

z_ind = 1

heatmap!(ax1, collect(interior(source_field))[:,:,z_ind], colormap=:viridis, colorrange=(minimum(source_field), maximum(source_field)))
Colorbar(fig1[1,2], label = "Tracer", vertical=true, colorrange=(minimum(source_field), maximum(source_field)))
heatmap!(ax2, reshape(sum(W_cons, dims = 2), Nx, Ny), colormap=:viridis, colorrange=(minimum(sum(W_cons, dims = 2)), maximum(sum(W_cons, dims = 2))))
Colorbar(fig1[1,4], label = "Weight", vertical=true, colorrange=(minimum(sum(W_cons, dims = 2)), maximum(sum(W_cons, dims = 2))))
heatmap!(ax3, (dst_vec), colormap=:viridis, colorrange=(minimum(dst_vec), maximum(dst_vec)))
Colorbar(fig1[1,6], label = "Tracer", vertical=true, colorrange=(minimum(dst_vec), maximum(dst_vec)))
# heatmap!(ax4, reshape(sum(W_bilin, dims = 2), Nx, Ny), colormap=:viridis, colorrange=(minimum(sum(W_bilin, dims = 2)), maximum(sum(W_bilin, dims = 2))))
# Colorbar(fig1[2,2], label = "Weight", vertical=true, colorrange=(minimum(sum(W_bilin, dims = 2)), maximum(sum(W_bilin, dims = 2))))
# heatmap!(ax5, collect(interior(dst_bilin))[:,:,z_ind], colormap=:viridis, colorrange=(minimum(dst_bilin), maximum(dst_bilin)))
# Colorbar(fig1[2,4], label = "Tracer", vertical=true, colorrange=(minimum(dst_bilin), maximum(dst_bilin)))

save(joinpath(figdir, "regrid_test_figure.png"), fig1, px_per_unit=3)

#=
"""
    regrid_tracers!(src::Field, dst::Field, W::SparseMatrixCSC)
Regrid the `src` field onto the `dst` field using the provided weights `W`.
The function assumes that the vertical grid (z) of both fields is the same.
"""
function regrid_tracers_test(dst::Field, src::Field, W)

    @assert dst.grid.z.cᵃᵃᶜ[1:dst.grid.Nz] == src.grid.z.cᵃᵃᶜ[1:src.grid.Nz] "Source and destination grids must have exactly the same vertical grid (z)."
    
    # Perform regridding
    for k in 1:dst.grid.Nz
        # Flatten the source field for regridding
        src_flat = vec(collect(interior(src))[:,:,k])  # shape (ncell, 1)

        # Regrid the source field to the destination grid
        dst_vec = reshape(W * src_flat, dst.grid.Nx, dst.grid.Ny)

        # Fill the destination field with the regridded values
        interior(dst)[:,:,k] .= dst_vec
    end

    return deepcopy(dst)
end

test_dst = regrid_tracers_test(destination_field, source_field, W_cons)
=#