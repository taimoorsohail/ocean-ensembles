using Oceananigans
using Oceananigans.Units
using ClimaOcean
includet("BasinMask.jl")
using .BasinMask

Nx = Integer(360/4)
Ny = Integer(180/4)
Nz = Integer(100/4)

arch = CPU()
z_faces = (-4000, 0)

underlying_grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (5, 5, 4),
                    first_pole_longitude = 70,
                    north_poles_latitude = 55)

@info "Defining bottom bathymetry"

@time bottom_height = regrid_bathymetry(underlying_grid;
                                    minimum_depth = 10,
                                    interpolation_passes = 75, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
                                    major_basins = 2)

# For this bathymetry at this horizontal resolution we need to manually open the Gibraltar strait.
# view(bottom_height, 102:103, 124, 1) .= -400

@info "Defining grid"

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)
                    
@info grid           

free_surface = SplitExplicitFreeSurface(grid; substeps=30)
model = HydrostaticFreeSurfaceModel(; grid, free_surface)


simulation = Simulation(model, Δt=0.001, stop_iteration=2000)

function integrate_tuple(outputs; volmask, dims, condition, suffix::AbstractString)
    for key in keys(outputs)
        @info key
        int_model_outputs = Integral(outputs[key]; dims, condition)
        # int_model_outputs = NamedTuple((Symbol(string(key) * suffix) => Integral(outputs[key]; dims, condition)))
    end
    @info "I am doing dV now"
    dV_int = NamedTuple{(Symbol(:dV, suffix),)}((Integral(volmask; dims, condition),))
    # @info "I am merging now"
    # int_outputs = merge(int_model_outputs, dV_int)
    return nothing #int_model_outputs#int_outputs
end

c = CenterField(grid)
volmask =  set!(c, 1)

@info "Defining masks"

Atlantic_mask = repeat(basin_mask(grid, "atlantic", c, arch), 1, 1, Nz)
IPac_mask = repeat(basin_mask(grid, "indo-pacific", c, arch), 1, 1, Nz)
glob_mask = Atlantic_mask .|| IPac_mask

@info size(Atlantic_mask), size(IPac_mask), size(glob_mask)

tracers = model.tracers
velocities = model.velocities

outputs = merge(tracers, velocities)

@time global_zonal_int_outputs = integrate_tuple(outputs; volmask, dims = (1), condition = glob_mask, suffix = "_global")
@time Atlantic_zonal_int_outputs = integrate_tuple(outputs; volmask, dims = (1), condition = Atlantic_mask, suffix = "_atlantic")
@time IPac_zonal_int_outputs = integrate_tuple(outputs; volmask, dims = (1), condition = IPac_mask, suffix = "_pacific")

# @time zonal_int_outputs = merge(global_zonal_int_outputs, Atlantic_zonal_int_outputs, IPac_zonal_int_outputs)
