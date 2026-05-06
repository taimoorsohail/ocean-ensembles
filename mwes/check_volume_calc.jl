using Oceananigans
using ClimaOcean

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")

arch = CPU()

Nx = 10
Ny = 10
Nz = 10

depth = -6000meters
z = ExponentialDiscretization(Nz, depth, 0; scale = depth/4)

underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

# Next, we build bathymetry on this grid, using interpolation passes to smooth the bathymetry.
# With 2 major basins, we keep the Mediterranean (though we need to manually open the Gibraltar
# Strait to connect it to the Atlantic):

ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir = data_path)

@time bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
                                minimum_depth = 15,
                                interpolation_passes = 1, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
                                major_basins = 4)


# We then incorporate the bathymetry into an ImmersedBoundaryGrid,

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                            active_cells_map=true)

V_ccc = KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.Vᶜᶜᶜ, grid)

totint_vol_c = Field(Integral(V_ccc, dims = (1,2,3)))
totint_vol_c_z = Field(Integral(V_ccc, dims = (1,2)))



@show totint_vol_c[1,1,1]

@show totint_vol_c_z[1,1,:]