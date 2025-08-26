using MPI
using CUDA

MPI.Init()
atexit(MPI.Finalize)  

using PyCall

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

using Oceananigans.Fields: location, ReducedField

const SomeTripolarGrid = Union{TripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}}
const SomeLatitudeLongitudeGrid = Union{LatitudeLongitudeGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid}}
const TripolarOrLatLonGrid = Union{SomeTripolarGrid, SomeLatitudeLongitudeGrid}


data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")

## Argument is provided by the submission script!

arch = CPU()

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


destination_field = Field{Center, Center, Center}(on_architecture(CPU(), destination_grid))
source_field = Field{Center, Center, Center}(on_architecture(CPU(), grid))

@info "Defining function"

    function regridder_weights_new(
    source_field, 
    destination_field; 
    method = "conservative")

        # Create source and destination fields

        dst = deepcopy(destination_field)
        src = deepcopy(source_field)

        @info "Extracting Centers"
        # Extract Centers
        if isa(source_field.grid, SomeTripolarGrid)
            src_lats = src.grid.φᶜᶜᵃ[1:src.grid.Nx, 1:src.grid.Ny]
            src_lons = src.grid.λᶜᶜᵃ[1:src.grid.Nx, 1:src.grid.Ny]
        elseif isa(source_field.grid, SomeLatitudeLongitudeGrid)
            src_lats = src.grid.φᵃᶜᵃ[1:src.grid.Ny]
            src_lons = src.grid.λᶜᵃᵃ[1:src.grid.Nx]
        else
            error("Unsupported grid type for source grid")
        end

        if isa(destination_field.grid, SomeTripolarGrid)
            dst_lats = dst.grid.φᶜᶜᵃ[1:dst.grid.Nx, 1:dst.grid.Ny]
            dst_lons = dst.grid.λᶜᶜᵃ[1:dst.grid.Nx, 1:dst.grid.Ny]
        elseif isa(destination_field.grid, SomeLatitudeLongitudeGrid)
            dst_lats = dst.grid.φᵃᶜᵃ[1:dst.grid.Ny]
            dst_lons = dst.grid.λᶜᵃᵃ[1:dst.grid.Nx]
        else
            error("Unsupported grid type for destination grid")
        end

        # Extract corners
        @info "Extracting Corners"

        if isa(source_field.grid, SomeTripolarGrid)
            src_lats_b = src.grid.φᶠᶠᵃ[1:src.grid.Nx+1, 1:src.grid.Ny+1]
            src_lons_b = src.grid.λᶠᶠᵃ[1:src.grid.Nx+1, 1:src.grid.Ny+1]
        elseif isa(source_field.grid, SomeLatitudeLongitudeGrid)
            src_lats_b = src.grid.φᵃᶠᵃ[1:src.grid.Ny+1]
            src_lons_b = src.grid.λᶠᵃᵃ[1:src.grid.Nx+1]
        else
            error("Unsupported grid type for source grid")
        end

        if isa(destination_field.grid, SomeTripolarGrid)
            dst_lats_b = dst.grid.φᶠᶠᵃ[1:dst.grid.Nx+1, 1:dst.grid.Ny+1]
            dst_lons_b = dst.grid.λᶠᶠᵃ[1:dst.grid.Nx+1, 1:dst.grid.Ny+1]
        elseif isa(destination_field.grid, SomeLatitudeLongitudeGrid)
            dst_lats_b = dst.grid.φᵃᶠᵃ[1:dst.grid.Ny+1]
            dst_lons_b = dst.grid.λᶠᵃᵃ[1:dst.grid.Nx+1]
        else
            error("Unsupported grid type for destination grid")
        end

        # Move to Python
        # Convert to numpy arrays for xesmf
        @info "Moving to Python"
        lat_src_np = OceanEnsembles.get_np().array(src_lats)
        lon_src_np = OceanEnsembles.get_np().array(src_lons)

        lat_dst_np = OceanEnsembles.get_np().array(dst_lats)
        lon_dst_np = OceanEnsembles.get_np().array(dst_lons)

        lat_src_b_np = OceanEnsembles.get_np().array(src_lats_b)
        lon_src_b_np = OceanEnsembles.get_np().array(src_lons_b)

        lat_dst_b_np = OceanEnsembles.get_np().array(dst_lats_b)
        lon_dst_b_np = OceanEnsembles.get_np().array(dst_lons_b)
        @info "creating xarray DataArrays"

        # Create xarray DataArrays for source and destination grids
        dst_data = Field{Center, Center, Nothing}(destination_field.grid)
        src_data = Field{Center, Center, Nothing}(source_field.grid)

        src_np = OceanEnsembles.get_np().squeeze(OceanEnsembles.get_np().array(collect(interior(src_data))))
        dst_np = OceanEnsembles.get_np().squeeze(OceanEnsembles.get_np().array(collect(interior(dst_data))))

        src_da = OceanEnsembles.get_xr().DataArray(
            src_np,
            dims=["y", "x"],
            coords= (Dict(
                "lat" => (["y", "x"], lat_src_np),
                "lon" => (["y", "x"], lon_src_np)
            )),
            name="source"
        )

        src_b_da = OceanEnsembles.get_xr().DataArray(
            lat_src_b_np,
            dims=["y_b", "x_b"],
            coords= (Dict(
                "lat_b" => (["y_b", "x_b"], lat_src_b_np),
                "lon_b" => (["y_b", "x_b"], lon_src_b_np)
            )),
            name="bounds"
        )

        src_ds = OceanEnsembles.get_xr().Dataset(
            (Dict("source" => src_da, "bounds" => src_b_da))
        )

        dst_ds = OceanEnsembles.get_xr().DataArray(
            dst_np,
            dims=["lon", "lat"],
            coords= (Dict(
                "lat" => (["lat"], lat_dst_np),
                "lon" => (["lon"], lon_dst_np)    )),
            name="destination"
        )

        @show src_ds, dst_ds
        @info "Creating regridder"
        regridder = OceanEnsembles.get_xesmf().Regridder(src_ds, dst_ds, method)

        # Move back to Julia
        @info "Moving array back to Julia"
        # Convert the regridder weights to a Julia sparse matrix
        coo = regridder.weights.data
        coords = coo[:coords]
        rows = coords[1,:].+1
        cols = coords[2,:].+1
        vals = Float64.(coo[:data])

        shape = Tuple(Int.(coo[:shape]))
        W = sparse(rows, cols, vals, shape[1], shape[2])

    return W
end

@info "Defining regridder weights"
@time W_bilin = regridder_weights_new(source_field, destination_field; method = "bilinear")
