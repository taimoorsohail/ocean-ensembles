
using GLMakie 
using Dates

using OceanEnsembles
using Oceananigans
using ClimaOcean

# ### Grid and Bathymetry
@info "Defining grid"
arch = CPU()
Nx = Integer(360)
Ny = Integer(180)
Nz = Integer(75)

@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
r_faces = ExponentialCoordinate(Nz, depth, 0)
# @show r_faces(Nz)
# r_faces = Oceananigans.Grids.MutableVerticalDiscretization(r_faces)

@info "Defining source underlying grid"

underlying_source_grid = TripolarGrid(arch;
                            size = (Nx, Ny, Nz),
                            z = r_faces,
                            halo = (5, 5, 4),
                            first_pole_longitude = 70,
                            north_poles_latitude = 55)

@info "Interpolating bottom bathymetry"

data_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles/data/")

ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir = data_path)
ClimaOcean.DataWrangling.download_dataset(ETOPOmetadata)

@time bottom_height = regrid_bathymetry(underlying_source_grid, ETOPOmetadata;
                                  minimum_depth = 10,
                                  interpolation_passes = 10, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
				                  major_basins = 2)
# view(bottom_height, 73:78, 88:89, 1) .= -1000 # open Gibraltar strait
                                  
@info "Defining source grid"

@time source_grid = ImmersedBoundaryGrid(underlying_source_grid, GridFittedBottom(bottom_height); active_cells_map=true)

@info "Defining destination underlying grid"
### DESTINATION FIELD
underlying_destination_grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             z = r_faces,
                             latitude  = (-80, 90),
                             longitude = (0, 360))

@info "Interpolating bottom bathymetry"

@time bottom_height = regrid_bathymetry(underlying_destination_grid, ETOPOmetadata;
                                  minimum_depth = 10,
                                  interpolation_passes = 10, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
				                  major_basins = 2)
# view(bottom_height, 73:78, 88:89, 1) .= -1000 # open Gibraltar strait
                                  
@info "Defining destination grid"

@time destination_grid = ImmersedBoundaryGrid(underlying_destination_grid, GridFittedBottom(bottom_height); active_cells_map=true)

source_field = Field{Center, Center, Center}(source_grid)
destination_field = Field{Center, Center, Center}(destination_grid)
# f(x, y, z) = y > 0 ? 1.0 : 0.0 # Example function to set the source field
# set!(source_field, f)

@info "Downloading/checking input data"

dates = vcat(collect(DateTime(1991, 1, 1): Month(1): DateTime(1991, 5, 1)),
             collect(DateTime(1990, 5, 1): Month(1): DateTime(1990, 12, 1)))

@info "We download the 1990-1991 data for an RYF implementation"

dataset = EN4Monthly() # Other options include ECCO2Monthly(), ECCO4Monthly() or ECCO2Daily()

temperature = Metadata(:temperature; dates, dataset = dataset, dir=data_path)
salinity    = Metadata(:salinity;    dates, dataset = dataset, dir=data_path)

t = FieldTimeSeries(temperature)   
s = FieldTimeSeries(salinity)

Tarr = t[1][:,:,42]
@info "Setting source field with temperature data"
set!(source_field, Tarr)

# Test the regridding functions
@time W_cons = regridder_weights!(source_field, destination_field; method = "conservative")
@time dst_cons = regrid_tracers!(source_field, destination_field, W_cons)
@time W_bilin = regridder_weights!(source_field, destination_field; method = "bilinear")
@time dst_bilin = regrid_tracers!(source_field, destination_field, W_bilin)

# Plotting the regridded data vs the source data
fig1 = Figure()
ax1 = Axis(fig1[1, 1], title="Source Field", xlabel="i", ylabel="j")
ax2 = Axis(fig1[1, 3], title="Regridding Weights - Conservative", xlabel="i", ylabel="j")
ax3 = Axis(fig1[1, 5], title="Destination Field - Conservative", xlabel="Longitude", ylabel="Latitude")
ax4 = Axis(fig1[2, 1], title="Regridding Weights - Bilinear", xlabel="Longitude", ylabel="Latitude")
ax5 = Axis(fig1[2, 3], title="Destination Field - Bilinear", xlabel="Longitude", ylabel="Latitude")

z_ind = 1

heatmap!(ax1, collect(interior(source_field))[:,:,z_ind], colormap=:viridis, colorrange=(minimum(source_field), maximum(source_field)))
Colorbar(fig1[1,2], label = "Tracer", vertical=true, colorrange=(minimum(source_field), maximum(source_field)))
heatmap!(ax2, reshape(sum(W_cons, dims = 2), Nx, Ny), colormap=:viridis, colorrange=(minimum(sum(W_cons, dims = 2)), maximum(sum(W_cons, dims = 2))))
Colorbar(fig1[1,4], label = "Weight", vertical=true, colorrange=(minimum(sum(W_cons, dims = 2)), maximum(sum(W_cons, dims = 2))))
heatmap!(ax3, collect(interior(dst_cons))[:,:,z_ind], colormap=:viridis, colorrange=(minimum(dst_cons), maximum(dst_cons)))
Colorbar(fig1[1,6], label = "Tracer", vertical=true, colorrange=(minimum(dst_cons), maximum(dst_cons)))
heatmap!(ax4, reshape(sum(W_bilin, dims = 2), Nx, Ny), colormap=:viridis, colorrange=(minimum(sum(W_bilin, dims = 2)), maximum(sum(W_bilin, dims = 2))))
Colorbar(fig1[2,2], label = "Weight", vertical=true, colorrange=(minimum(sum(W_bilin, dims = 2)), maximum(sum(W_bilin, dims = 2))))
heatmap!(ax5, collect(interior(dst_bilin))[:,:,z_ind], colormap=:viridis, colorrange=(minimum(dst_bilin), maximum(dst_bilin)))
Colorbar(fig1[2,4], label = "Tracer", vertical=true, colorrange=(minimum(dst_bilin), maximum(dst_bilin)))

display(fig1)

# fig = Figure()
# ax1 = Axis(fig[1, 1], title="Source Areas", xlabel="Point #", ylabel="Area")
# ax2 = Axis(fig[3, 1], title="Dest. Areas", xlabel="Point #", ylabel="Area")

# heatmap!(ax1, reshape(row_sums, 360,180), colormap=:viridis, colorrange=(0, maximum(row_sums)))
# Colorbar(fig[2,1], label = "Source Temperature", vertical=false, colorrange=(0, maximum(row_sums)))

# heatmap!(ax2, (reshape(W_sums, 360,180))  , colorrange=(0, maximum(W_sums)))
# Colorbar(fig[4,1], label = "Dest. Temperature", vertical=false, colorrange=(0, maximum(W_sums)))

# display(fig)
