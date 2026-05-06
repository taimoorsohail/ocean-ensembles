using Oceananigans
using Oceananigans.Architectures: architecture
using NumericalEarth
using NumericalEarth.DataWrangling.ETOPO
using OceanEnsembles
using CairoMakie

# Single-rank bathymetry mask MWE.
# Builds original bathymetry, applies the regional masks, and plots both fields.

const data_path = get(ENV, "OCEAN_ENSEMBLES_DATA_PATH", "/g/data/v46/txs156/ocean-ensembles/data/")
const figdir = get(ENV, "OCEAN_ENSEMBLES_FIG_PATH", joinpath(@__DIR__, "..", "figures"))

mkpath(figdir)

arch = CPU() # single rank MWE

Nx = 360 * 6
Ny = 180 * 6
Nz = 75

@info "Constructing grid" arch Nx Ny Nz

z_faces = ExponentialDiscretization(Nz, -6000.0, 0; mutable=true)
underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), z=z_faces, halo=(7, 7, 7))

@info "Loading ETOPO and regridding bathymetry"
ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir=data_path)
NumericalEarth.DataWrangling.download_dataset(ETOPOmetadata)

bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
                                  minimum_depth=15,
                                  interpolation_passes=25,
                                  major_basins=4)

bh_original = Array(interior(bottom_height)[:, :, 1])
bh_masked = copy(bh_original)

@info "Applying single-rank bathymetry masks"

# Black + Caspian Seas
xs_black_caspian = [755, 1010, 1010, 755]
ys_black_caspian = [800, 790, 920, 920]
mask_black_caspian = section_mask(xs_black_caspian, ys_black_caspian,
                                  fill(1, length(xs_black_caspian)), underlying_grid)

idx_black_caspian = (mask_black_caspian .== 1) .& (bh_masked .<= 0)
bh_masked[idx_black_caspian] .= 0

# Baltic Sea / Danish Straits
xs_danish = [679, 670 - 9, 679, 688 + 9]
ys_danish = [872 - 10, 875 + 6, 882 + 5, 875 + 6]
mask_danish = section_mask(xs_danish, ys_danish,
                           fill(2, length(xs_danish)), underlying_grid)

idx_danish = (mask_danish .== 2) .& (bh_masked .>= 0) .& (bh_masked .<= 3)
bh_masked[idx_danish] .= -10

# Shared color range so visual comparison is meaningful.
vmin = min(minimum(bh_original), minimum(bh_masked))
vmax = max(maximum(bh_original), maximum(bh_masked))

fig = Figure(size=(2200, 1000))
ax1 = Axis(fig[1, 1], title="Original bathymetry", xlabel="x index", ylabel="y index")
ax2 = Axis(fig[1, 2], title="Masked bathymetry", xlabel="x index", ylabel="y index")

hm1 = heatmap!(ax1, bh_original; colormap=Reverse(:deep), colorrange=(vmin, vmax))
hm2 = heatmap!(ax2, bh_masked; colormap=Reverse(:deep), colorrange=(vmin, vmax))

Colorbar(fig[2, 1:2], hm1; label="bottom height")

output_file = joinpath(figdir, "bathymetry_single_rank_original_vs_masked.png")
save(output_file, fig, px_per_unit=1)

@info "Saved figure" output_file
