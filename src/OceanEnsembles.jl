module OceanEnsembles

export basin_mask, ocean_tracer_content!, volume_transport!, combine_outputs, identify_combination_targets

using ClimaOcean, Oceananigans, Glob

include("BasinMask.jl")
include("Diagnostics.jl")
include("OutputWrangling.jl")

using .BasinMask
using .Diagnostics
using .OutputWrangling

# using PrecompileTools: @setup_workload, @compile_workload

# @setup_workload begin
#     Nx = Integer(360/3)
#     Ny = Integer(180/3)
#     Nz = Integer(75)
        
#     @compile_workload begin
#         arch = CPU()
#         r_faces = ClimaOcean.exponential_z_faces(; Nz, depth=5000, h=12.43)
#         underlying_grid = Oceananigans.OrthogonalSphericalShellGrids.TripolarGrid(arch;
#                                         size = (Nx, Ny, Nz),
#                                         z = r_faces,
#                                         halo = (7, 7, 4),
#                                         first_pole_longitude = 70,
#                                         north_poles_latitude = 55)
#         data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
        
#         ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir = data_path)

#         bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
#                                           minimum_depth = 15,
#                                           major_basins = 1)

#         grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

#         ocean = ocean_simulation(grid; free_surface = SplitExplicitFreeSurface(grid; substeps=70))

#         radiation = Radiation(arch)

#         atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(25))

#         coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

#     end
# end

end # module OceanEnsembles
