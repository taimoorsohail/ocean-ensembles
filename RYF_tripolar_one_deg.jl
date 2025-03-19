using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
using ClimaOcean.ECCO
using Oceananigans.AbstractOperations
using BasinMask: basin_mask # This is the basin masking function
using Oceananigans.AbstractOperations: Integral

arch = CPU()

Nx, Ny, Nz = 60, 30, 20

grid = Oceananigans.OrthogonalSphericalShellGrids.TripolarGrid(arch;
                                                               size=(Nx, Ny, Nz),
                                                               halo=(7, 7, 7),
                                                               z=(-6000,0))

ocean = ocean_simulation(grid)

date = DateTimeProlepticGregorian(1993, 1, 1)
set!(ocean.model, T=Metadata(:temperature; dates=date, dataset=ECCO4Monthly()),
                  S=Metadata(:salinity; dates=date, dataset=ECCO4Monthly()))

radiation = Radiation(arch)

atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(41))

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

simulation = Simulation(coupled_model; Δt=30minutes, stop_time=50days)

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T

    Tmax = maximum(interior(T))
    Tmin = minimum(interior(T))

    umax = (maximum(abs, interior(u)),
            maximum(abs, interior(v)),
            maximum(abs, interior(w)))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg = @sprintf("Iter: %d, simulation time: %s, atmosphere time: %s, Δt: %s", iteration(sim), prettytime(sim), prettytime(atmosphere.clock.time), prettytime(sim.Δt))
    msg *= @sprintf(", max|u|: (%.2e, %.2e, %.2e) m s⁻¹, extrema(T): (%.2f, %.2f) ᵒC, wall time: %s",
                    umax..., Tmax, Tmin, prettytime(step_time))

    @info msg

    wall_time[] = time_ns()
end

simulation.callbacks[:progress] = Callback(progress, TimeInterval(1days))


function average_tuple(ocean; volmask, dims::Tuple, conditions)
    avg_tracer_outputs = NamedTuple((key => Average(ocean.model.tracers[key], dims=dims, condition=conditions) for key in keys(ocean.model.tracers)))
    avg_velocities_outputs = NamedTuple((key => Average(ocean.model.velocities[key], dims=dims, condition=conditions) for key in keys(ocean.model.velocities)))
    dV_avg = NamedTuple{(:dV,)}((dV = Average(volmask, dims=dims, condition=conditions),))
    avg_outputs = merge(avg_tracer_outputs, avg_velocities_outputs, dV_avg)
    return avg_outputs
end

function integrate_tuple(ocean;  volmask, dims::Tuple, conditions)
    int_tracer_outputs = NamedTuple((key => Integral(ocean.model.tracers[key], dims=dims, condition=conditions) for key in keys(ocean.model.tracers)))
    int_velocities_outputs = NamedTuple((key => Integral(ocean.model.velocities[key], dims=dims, condition=conditions) for key in keys(ocean.model.velocities)))
    dV_int = NamedTuple{(:dV,)}((dV = Integral(volmask, dims=dims, condition=conditions),))
    int_outputs = merge(int_tracer_outputs, int_velocities_outputs, dV_int)
    return int_outputs
end

c = CenterField(grid)
volmask =  set!(c, 1)

Global_mask = basin_mask(grid, "", ocean.model.tracers.T)
Atlantic_mask = basin_mask(grid, "atlantic", ocean.model.tracers.T)
IPac_mask = basin_mask(grid, "indo-pacific", ocean.model.tracers.T)

#### SURFACE
# ASK: how to integrate a boolean mask with a surface field?
tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)

#### AVERAGING
# Save NamedTuples of averaged tracers
global_avg_outputs = average_tuple(ocean; volmask, dims = (1,2,3), conditions = Global_mask)
Atlantic_avg_outputs = average_tuple(ocean; volmask, dims = (1,2,3), conditions = Atlantic_mask)
IPac_avg_outputs = average_tuple(ocean; volmask, dims = (1,2,3), conditions = IPac_mask)

global_depth_avg_outputs = average_tuple(ocean; volmask, dims = (1,2), Global_mask)
Atlantic_depth_avg_outputs = average_tuple(ocean; volmask, dims = (1,2), Atlantic_mask)
IPac_depth_avg_outputs = average_tuple(ocean; volmask, dims = (1,2), IPac_mask)

global_zonal_avg_outputs = average_tuple(ocean; volmask, dims = (1), Global_mask)
Atlantic_zonal_avg_outputs = average_tuple(ocean; volmask, dims = (1), Atlantic_mask)
IPac_zonal_avg_outputs = average_tuple(ocean; volmask, dims = (1), IPac_mask)

#### INTEGRATING
global_int_outputs = integrate_tuple(ocean; volmask, dims = (1,2,3), Global_mask)
Atlantic_int_outputs = integrate_tuple(ocean; volmask, dims = (1,2,3), Atlantic_mask)
IPac_int_outputs = integrate_tuple(ocean; volmask, dims = (1,2,3), IPac_mask)

global_depth_int_outputs = integrate_tuple(ocean; volmask, dims = (1,2), Global_mask)
Atlantic_depth_int_outputs = integrate_tuple(ocean; volmask, dims = (1,2), Atlantic_mask)
IPac_depth_int_outputs = integrate_tuple(ocean; volmask, dims = (1,2), IPac_mask)

global_zonal_int_outputs = integrate_tuple(ocean; volmask, dims = (1), Global_mask)
Atlantic_zonal_int_outputs = integrate_tuple(ocean; volmask, dims = (1), Atlantic_mask)
IPac_zonal_int_outputs = integrate_tuple(ocean; volmask, dims = (1), IPac_mask)

## TODO - turn this into nested tuples too? Make it more efficient?

# avg_tracer_outputs = NamedTuple((key => Average(tracers[key]) for key in keys(tracers)))
# Save NamedTuples of depth averaged tracers & velocities
# depth_avg_velocity_outputs = NamedTuple((key => Average(velocities[key], dims=(1,2)) for key in keys(velocities)))
# depth_avg_tracer_outputs = NamedTuple((key => Average(tracers[key], dims=(1,2)) for key in keys(tracers)))
# depth_avg_outputs = merge(depth_avg_tracer_outputs, depth_avg_velocity_outputs)
# # Save NamedTuples of zonally-averaged tracers & velocities
# zonal_avg_velocity_outputs = NamedTuple((key => Average(velocities[key], dims=1) for key in keys(velocities)))
# zonal_avg_tracer_outputs = NamedTuple((key => Average(tracers[key], dims=1) for key in keys(tracers)))
# zonal_avg_outputs = merge(zonal_avg_tracer_outputs, zonal_avg_velocity_outputs)

# #### INTEGRATING
# # Save NamedTuples of depth integrated tracers
# dV_tuple_depth_avg = NamedTuple{(:dV,)}((dV = Average(volmask, dims=(1,2)),))
# dV_tuple_zonal_avg = NamedTuple{(:dV,)}((dV = Average(volmask, dims=1),))

# dV_tuple_depth = NamedTuple{(:dV,)}((dV = Integral(volmask, dims=(1,2)),))
# dV_tuple_zonal = NamedTuple{(:dV,)}((dV = Integral(volmask, dims=1),))

# depth_int_tracer_outputs = merge(
#     NamedTuple((key => Integral(tracers[key]; dims=(1,2))) for key in keys(tracers)),
#     dV_tuple_depth)
# # Save NamedTuples of zonally integrated tracers
# zonal_int_tracer_outputs = merge(
#     NamedTuple((key => Integral(tracers[key]; dims=1)) for key in keys(tracers)),
#     dV_tuple_zonal)

# TODO: Make these constants saved as well for OHC; OSC
ρₒ = simulation.model.interfaces.ocean_properties.reference_density
cₚ = simulation.model.interfaces.ocean_properties.heat_capacity
S₀ = 35 #g/kg

constants = NamedTuple{(:reference_density, :heat_capacity, :reference_salinity)}((ρₒ, cₚ, S₀))

simulation.output_writers[:surface] = JLD2Outpur(ocean.model, outputs;
                                                 schedule = TimeInterval(5days),
                                                 filename = "global_surface_fields",
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = true,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

simulation.output_writers[:global_avg] = JLD2Output(ocean.model, avg_tracer_outputs;
                                                    schedule = TimeInterval(1days),
                                                    filename = "averaged_data",
                                                    overwrite_existing = true)

simulation.output_writers[:global_depth_avg] = JLD2Output(ocean.model, depth_avg_outputs;
                                                          schedule = TimeInterval(1days),
                                                          filename = "depth_averaged_data",
                                                          overwrite_existing = true)

simulation.output_writers[:global_zonal_avg] = JLD2Output(ocean.model, zonal_avg_outputs;
                                                          schedule = TimeInterval(1days),
                                                          filename = "zonal_averaged_data",
                                                          overwrite_existing = true)

simulation.output_writers[:global_depth_int] = JLD2Output(ocean.model, depth_int_tracer_outputs;
                                                          schedule = TimeInterval(1days),
                                                          filename = "depth_integrated_data",
                                                          overwrite_existing = true)

simulation.output_writers[:global_zonal_int] = JLD2Output(ocean.model, zonal_int_tracer_outputs;
                                                          schedule = TimeInterval(1days),
                                                          filename = "zonal_integrated_data",
                                                          overwrite_existing = true)

# simulation.output_writers[:constants] = JLD2Output(ocean.model, constants;
#                                                    schedule = TimeInterval(1days),
#                                                    filename = "constants",
#                                                    overwrite_existing = true)

run!(simulation)