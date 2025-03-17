using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
using ClimaOcean.ECCO
using Oceananigans.AbstractOperations

arch = CPU()

Nx = 144
Ny = 60
Nz = 40

depth = 6000meters
z_faces = exponential_z_faces(; Nz, depth)

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             z = z_faces,
                             latitude  = (-75, 75),
                             longitude = (0, 360))

ocean = ocean_simulation(grid)

date = DateTimeProlepticGregorian(1993, 1, 1)
set!(ocean.model, T=Metadata(:temperature; dates=date, dataset=ECCO4Monthly()),
                  S=Metadata(:salinity; dates=date, dataset=ECCO4Monthly()))

radiation = Radiation(arch)

atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(41))

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

simulation = Simulation(coupled_model; Δt=30minutes, stop_time=10days)

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

outputs = merge(ocean.model.tracers, ocean.model.velocities)

#### AVERAGING
# Save NamedTuple of Global Averaged tracers
avg_tracer_outputs = NamedTuple((key => Average(ocean.model.tracers[key]) for key in keys(ocean.model.tracers)))
# Save NamedTuples of depth averaged tracers & velocities
depth_avg_velocity_outputs = NamedTuple((key => Average(ocean.model.velocities[key], dims=(1,2)) for key in keys(ocean.model.velocities)))
depth_avg_tracer_outputs = NamedTuple((key => Average(ocean.model.tracers[key], dims=(1,2)) for key in keys(ocean.model.tracers)))
depth_avg_outputs = merge(depth_avg_tracer_outputs, depth_avg_velocity_outputs)
# Save NamedTuples of zonally-averaged tracers & velocities
zonal_avg_velocity_outputs = NamedTuple((key => Average(ocean.model.velocities[key], dims=1) for key in keys(ocean.model.velocities)))
zonal_avg_tracer_outputs = NamedTuple((key => Average(ocean.model.tracers[key], dims=1) for key in keys(ocean.model.tracers)))
zonal_avg_outputs = merge(zonal_avg_tracer_outputs, zonal_avg_velocity_outputs)

#### INTEGRATING
# Save NamedTuples of depth integrated tracers
c = CenterField(grid)
volmask =  set!(c, 1)
dV_tuple_depth = NamedTuple{(:dV,)}((dV = Oceananigans.AbstractOperations.Integral(volmask, dims=(1,2)),))
dV_tuple_zonal = NamedTuple{(:dV,)}((dV = Oceananigans.AbstractOperations.Integral(volmask, dims=1),))

depth_int_tracer_outputs = merge(
    NamedTuple((key => Oceananigans.AbstractOperations.Integral(ocean.model.tracers[key]; dims=(1,2))) for key in keys(ocean.model.tracers)),
    dV_tuple_depth)
# Save NamedTuples of zonally integrated tracers
zonal_int_tracer_outputs = merge(
    NamedTuple((key => Oceananigans.AbstractOperations.Integral(ocean.model.tracers[key]; dims=1)) for key in keys(ocean.model.tracers)),
    dV_tuple_zonal)

ρₒ = simulation.model.interfaces.ocean_properties.reference_density
cₚ = simulation.model.interfaces.ocean_properties.heat_capacity
S₀ = 35 #g/kg

constants = NamedTuple{(:reference_density, :heat_capacity, :reference_salinity)}((ρₒ, cₚ, S₀))

simulation.output_writers[:surface] = JLD2OutputWriter(ocean.model, outputs;
                                                  schedule = TimeInterval(5days),
                                                  filename = "global_surface_fields",
                                                  indices = (:, :, grid.Nz),
                                                  with_halos = true,
                                                  overwrite_existing = true,
                                                  array_type = Array{Float32})

simulation.output_writers[:global_avg] = JLD2OutputWriter(ocean.model, avg_tracer_outputs;
                                                  schedule = TimeInterval(1days),
                                                  filename = "averaged_tracer_data",
                                                  overwrite_existing = true)

simulation.output_writers[:global_depth_avg] = JLD2OutputWriter(ocean.model, depth_avg_tracer_outputs;
                                                  schedule = TimeInterval(1days),
                                                  filename = "depth_averaged_tracer_data",
                                                  overwrite_existing = true)

simulation.output_writers[:global_zonal_avg] = JLD2OutputWriter(ocean.model, zonal_avg_tracer_outputs;
                                                  schedule = TimeInterval(1days),
                                                  filename = "zonal_averaged_tracer_data",
                                                  overwrite_existing = true)

simulation.output_writers[:global_depth_int] = JLD2OutputWriter(ocean.model, depth_int_tracer_outputs;
                                                  schedule = TimeInterval(1days),
                                                  filename = "depth_integrated_tracer_data",
                                                  overwrite_existing = true)

simulation.output_writers[:global_zonal_int] = JLD2OutputWriter(ocean.model, zonal_int_tracer_outputs;
                                                  schedule = TimeInterval(1days),
                                                  filename = "zonal_integrated_tracer_data",
                                                  overwrite_existing = true)

simulation.output_writers[:constants] = JLD2OutputWriter(ocean.model, constants;
                                                  schedule = TimeInterval(1days),
                                                  filename = "constants",
                                                  overwrite_existing = true)

run!(simulation)