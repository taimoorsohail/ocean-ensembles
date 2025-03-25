using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
includet("BasinMask.jl")
using .BasinMask

arch = CPU()

Nx = Integer(360/4)
Ny = Integer(180/4)
Nz = Integer(100/4)

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

ocean = simulation.model.ocean
atmosphere = simulation.model.atmosphere

#### We define our named tuple outputs

## Use ClimaOcean checkpointer branch
@info "Defining averaging functions"

function average_tuple(outputs; volmask, dims, condition=convert(Array{Bool}, ones(Ny, Nx, Nz)), suffix::AbstractString)
    avg_model_outputs = NamedTuple((Symbol(string(key) * suffix) => Average(outputs[key]; dims, condition) for key in keys(outputs)))
    dV_avg = NamedTuple{(Symbol(:dV, suffix),)}((Average(volmask; dims, condition),))
    avg_outputs = merge(avg_model_outputs, dV_avg)
    return avg_outputs
end

function integrate_tuple(outputs; volmask, dims, condition=convert(Array{Bool}, ones(Ny, Nx, Nz)), suffix::AbstractString) # Add suffix kwarg
    int_model_outputs = NamedTuple((Symbol(string(key) * suffix) => Integral(outputs[key]; dims, condition) for key in keys(outputs)))
    dV_int = NamedTuple{(Symbol(:dV, suffix),)}((Integral(volmask; dims, condition),))
    int_outputs = merge(int_model_outputs, dV_int)
    return int_outputs
end

c = CenterField(grid)
volmask =  set!(c, 1)

@info "Defining masks"

Atlantic_mask = repeat(basin_mask(grid, "atlantic", c), 1, 1, Nz)
IPac_mask = repeat(basin_mask(grid, "indo-pacific", c), 1, 1, Nz)

#### SURFACE

@info "Defining surface outputs"

tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)

#### AVERAGING
# Save NamedTuples of averaged tracers

# @info "Defining averaged outputs"

# @time global_avg_outputs = average_tuple(outputs; volmask, dims = (1,2,3), suffix = "_global")
# @time Atlantic_avg_outputs = average_tuple(outputs; volmask, dims = (1,2,3), condition = Atlantic_mask, suffix = "_atlantic")
# @time IPac_avg_outputs = average_tuple(outputs; volmask, dims = (1,2,3), condition = IPac_mask, suffix = "_pacific")

# @time global_depth_avg_outputs = average_tuple(outputs; volmask, dims = (1,2), suffix = "_global")
# @time Atlantic_depth_avg_outputs = average_tuple(outputs; volmask, dims = (1,2), condition = Atlantic_mask, suffix = "_atlantic")
# @time IPac_depth_avg_outputs = average_tuple(outputs; volmask, dims = (1,2), condition = IPac_mask, suffix = "_pacific")

# @time global_zonal_avg_outputs = average_tuple(outputs; volmask, dims = (1), suffix = "_global")
# @time Atlantic_zonal_avg_outputs = average_tuple(outputs; volmask, dims = (1), condition = Atlantic_mask, suffix = "_atlantic")
# @time IPac_zonal_avg_outputs = average_tuple(outputs; volmask, dims = (1), condition = IPac_mask, suffix = "_pacific")

# @time avg_outputs = merge(global_avg_outputs, Atlantic_avg_outputs, IPac_avg_outputs)
# @time depth_avg_outputs = merge(global_depth_avg_outputs, Atlantic_depth_avg_outputs, IPac_depth_avg_outputs)
# @time zonal_avg_outputs = merge(global_zonal_avg_outputs, Atlantic_zonal_avg_outputs, IPac_zonal_avg_outputs)

#### INTEGRATING

@info "Defining integrated outputs"

# @time global_int_outputs = integrate_tuple(outputs; volmask, dims = (1,2,3), suffix = "_global")
# @time Atlantic_int_outputs = integrate_tuple(outputs; volmask, dims = (1,2,3), condition = Atlantic_mask, suffix = "_atlantic")
# @time IPac_int_outputs = integrate_tuple(outputs; volmask, dims = (1,2,3), condition = IPac_mask, suffix = "_pacific")

# @time global_depth_int_outputs = integrate_tuple(outputs; volmask, dims = (1,2), suffix = "_global")
# @time Atlantic_depth_int_outputs = integrate_tuple(outputs; volmask, dims = (1,2), condition = Atlantic_mask, suffix = "_atlantic")
# @time IPac_depth_int_outputs = integrate_tuple(outputs; volmask, dims = (1,2), condition = IPac_mask, suffix = "_pacific")

@info "In theory, we only need the integrated zonal tracers + int(dV) to compute everything!"

@time global_zonal_int_outputs = integrate_tuple(outputs; volmask, dims = (1), suffix = "_global")
@time Atlantic_zonal_int_outputs = integrate_tuple(outputs; volmask, dims = (1), condition = Atlantic_mask, suffix = "_atlantic")
@time IPac_zonal_int_outputs = integrate_tuple(outputs; volmask, dims = (1), condition = IPac_mask, suffix = "_pacific")

# @time int_outputs = merge(global_int_outputs, Atlantic_int_outputs, IPac_int_outputs)
# @time depth_int_outputs = merge(global_depth_int_outputs, Atlantic_depth_int_outputs, IPac_depth_int_outputs)
@time zonal_int_outputs = merge(global_zonal_int_outputs, Atlantic_zonal_int_outputs, IPac_zonal_int_outputs)

constants = simulation.model.interfaces.ocean_properties

@info "Defining output writers"

simulation.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                                 schedule = TimeInterval(5days),
                                                 filename = "global_surface_fields",
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

# simulation.output_writers[:global_avg] = JLD2Writer(ocean.model, avg_tracer_outputs;
#                                                     schedule = TimeInterval(1days),
#                                                     filename = "averaged_data",
#                                                     overwrite_existing = true)

# simulation.output_writers[:global_depth_avg] = JLD2Writer(ocean.model, depth_avg_outputs;
#                                                           schedule = TimeInterval(1days),
#                                                           filename = "depth_averaged_data",
#                                                           overwrite_existing = true)

# simulation.output_writers[:global_zonal_avg] = JLD2Writer(ocean.model, zonal_avg_outputs;
#                                                           schedule = TimeInterval(1days),
#                                                           filename = "zonal_averaged_data",
#                                                           overwrite_existing = true)

# simulation.output_writers[:global_depth_int] = JLD2Writer(ocean.model, depth_int_tracer_outputs;
#                                                           schedule = TimeInterval(1days),
#                                                           filename = "depth_integrated_data",
#                                                           overwrite_existing = true)

simulation.output_writers[:global_zonal_int] = JLD2Writer(ocean.model, zonal_int_tracer_outputs;
                                                          schedule = TimeInterval(1days),
                                                          filename = "zonal_integrated_data",
                                                          overwrite_existing = true)

simulation.output_writers[:constants] = JLD2Writer(ocean.model, constants;
                                                   schedule = TimeInterval(365days),
                                                   filename = "constants",
                                                   overwrite_existing = true)

@info "Running simulation"

run!(simulation)
