using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
includet("BasinMask.jl")
using .BasinMask

arch = CPU()

Nx = Integer(360/3)
Ny = Integer(180/3)
Nz = Integer(100/2)

### We define our target grid to test tripolar, immersed boundary and mutable vertical grid

r_faces = exponential_z_faces(; Nz, depth=5000, h=34)
z_faces = Oceananigans.MutableVerticalDiscretization(r_faces)

underlying_grid = TripolarGrid(arch;
                               size = (Nx, Ny, Nz),
                               z = z_faces,
                               halo = (7, 7, 3),
                               first_pole_longitude = 70,
                               north_poles_latitude = 55)

bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 75, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
				                  major_basins = 2)

# For this bathymetry at this horizontal resolution we need to manually open the Gibraltar strait.
# view(bottom_height, 102:103, 124, 1) .= -400
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

#### We define our simulation

function testbed_coupled_simulation(arch, grid; stop_iteration=8)
    ocean = ocean_simulation(grid)

    radiation = Radiation(arch)

    atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(4))

    coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

    return Simulation(coupled_model; Δt=10, stop_iteration)
end

simulation = testbed_coupled_simulation(arch, grid; stop_iteration=8)

#### We define our callbacks

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    atmosphere = sim.model.atmosphere

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg = @sprintf("iteration: %d, sim time: %s, atmos time: %s, ocean time: %s, wall time: %s",
                   iteration(sim), sim.model.clock.time, atmosphere.clock.time, ocean.model.clock.time, prettytime(step_time))

    @info msg

    wall_time[] = time_ns()
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

#### We define our named tuple outputs

function average_tuple(outputs; volmask, dims, condition=convert(Array{Bool}, ones(Ny, Nx)), suffix::AbstractString)
    avg_model_outputs = NamedTuple((Symbol(string(key) * suffix) => Average(outputs[key]; dims, condition) for key in keys(outputs)))
    dV_avg = NamedTuple{(Symbol(:dV, suffix),)}((Average(volmask; dims, condition),))
    avg_outputs = merge(avg_model_outputs, dV_avg)
    return avg_outputs
end

function integrate_tuple(outputs; volmask, dims, condition=convert(Array{Bool}, ones(Ny, Nx)), suffix::AbstractString) # Add suffix kwarg
    int_model_outputs = NamedTuple((Symbol(string(key) * suffix) => Integral(outputs[key]; dims, condition) for key in keys(outputs)))
    dV_int = NamedTuple{(Symbol(:dV, suffix),)}((Integral(volmask; dims, condition),))
    int_outputs = merge(int_model_outputs, dV_int)
    return int_outputs
end

c = CenterField(grid)
volmask =  set!(c, 1)

#### And the basin masks

Atlantic_mask = repeat(basin_mask(grid, "atlantic", c), 1, 1, Nz)
IPac_mask = repeat(basin_mask(grid, "indo-pacific", c), 1, 1, Nz)

#### SURFACE
tracers = simulation.model.ocean.model.tracers
velocities = simulation.model.ocean.model.velocities

outputs = merge(tracers, velocities)

#### AVERAGING
# Save NamedTuples of averaged tracers
global_avg_outputs = average_tuple(outputs; volmask, dims = (1,2,3), suffix = "_global")
Atlantic_avg_outputs = average_tuple(outputs; volmask, dims = (1,2,3), condition = Atlantic_mask, suffix = "_atlantic")
IPac_avg_outputs = average_tuple(outputs; volmask, dims = (1,2,3), condition = IPac_mask, suffix = "_pacific")

global_depth_avg_outputs = average_tuple(outputs; volmask, dims = (1,2), suffix = "_global")
Atlantic_depth_avg_outputs = average_tuple(outputs; volmask, dims = (1,2), condition = Atlantic_mask, suffix = "_atlantic")
IPac_depth_avg_outputs = average_tuple(outputs; volmask, dims = (1,2), condition = IPac_mask, suffix = "_pacific")

global_zonal_avg_outputs = average_tuple(outputs; volmask, dims = (1), suffix = "_global")
Atlantic_zonal_avg_outputs = average_tuple(outputs; volmask, dims = (1), condition = Atlantic_mask, suffix = "_atlantic")
IPac_zonal_avg_outputs = average_tuple(outputs; volmask, dims = (1), condition = IPac_mask, suffix = "_pacific")

avg_outputs = merge(global_avg_outputs, Atlantic_avg_outputs, IPac_avg_outputs)
depth_avg_outputs = merge(global_depth_avg_outputs, Atlantic_depth_avg_outputs, IPac_depth_avg_outputs)
zonal_avg_outputs = merge(global_zonal_avg_outputs, Atlantic_zonal_avg_outputs, IPac_zonal_avg_outputs)

#### INTEGRATING
global_int_outputs = integrate_tuple(outputs; volmask, dims = (1,2,3), suffix = "_global")
Atlantic_int_outputs = integrate_tuple(outputs; volmask, dims = (1,2,3), condition = Atlantic_mask, suffix = "_atlantic")
IPac_int_outputs = integrate_tuple(outputs; volmask, dims = (1,2,3), condition = IPac_mask, suffix = "_pacific")

global_depth_int_outputs = integrate_tuple(outputs; volmask, dims = (1,2), suffix = "_global")
Atlantic_depth_int_outputs = integrate_tuple(outputs; volmask, dims = (1,2), condition = Atlantic_mask, suffix = "_atlantic")
IPac_depth_int_outputs = integrate_tuple(outputs; volmask, dims = (1,2), condition = IPac_mask, suffix = "_pacific")

global_zonal_int_outputs = integrate_tuple(outputs; volmask, dims = (1), suffix = "_global")
Atlantic_zonal_int_outputs = integrate_tuple(outputs; volmask, dims = (1), condition = Atlantic_mask, suffix = "_atlantic")
IPac_zonal_int_outputs = integrate_tuple(outputs; volmask, dims = (1), condition = IPac_mask, suffix = "_pacific")

int_outputs = merge(global_int_outputs, Atlantic_int_outputs, IPac_int_outputs)
depth_int_outputs = merge(global_depth_int_outputs, Atlantic_depth_int_outputs, IPac_depth_int_outputs)
zonal_int_outputs = merge(global_zonal_int_outputs, Atlantic_zonal_int_outputs, IPac_zonal_int_outputs)

output_iter = IterationInterval(1)

simulation.output_writers[:surface] = JLD2Output(ocean.model, outputs;
                                                 schedule = output_iter,
                                                 filename = "global_surface_fields",
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = true,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

simulation.output_writers[:global_avg] = JLD2Output(ocean.model, avg_tracer_outputs;
                                                    schedule = output_iter,
                                                    filename = "averaged_data",
                                                    overwrite_existing = true)

simulation.output_writers[:global_depth_avg] = JLD2Output(ocean.model, depth_avg_outputs;
                                                          schedule = output_iter,
                                                          filename = "depth_averaged_data",
                                                          overwrite_existing = true)

simulation.output_writers[:global_zonal_avg] = JLD2Output(ocean.model, zonal_avg_outputs;
                                                          schedule = output_iter,
                                                          filename = "zonal_averaged_data",
                                                          overwrite_existing = true)

simulation.output_writers[:global_depth_int] = JLD2Output(ocean.model, depth_int_tracer_outputs;
                                                          schedule = output_iter,
                                                          filename = "depth_integrated_data",
                                                          overwrite_existing = true)

simulation.output_writers[:global_zonal_int] = JLD2Output(ocean.model, zonal_int_tracer_outputs;
                                                          schedule = output_iter,
                                                          filename = "zonal_integrated_data",
                                                          overwrite_existing = true)

# simulation.output_writers[:checkpoint] = Checkpointer(ocean.model;
#                                                       schedule = output_iter,
#                                                       prefix = "checkpointer",
#                                                       dir = ".",
#                                                       verbose = true,
#                                                       overwrite_existing = true)

run!(simulation)