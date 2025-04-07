using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
using OceanEnsembles: basin_mask, ocean_tracer_content, volume_transport
using Oceananigans.Operators: Ax, Ay, Az, Δz

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
view(bottom_height, 102:103, 124, 1) .= -400

@info "Defining grid"

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

@info "Defining free surface"

free_surface = SplitExplicitFreeSurface(grid; substeps=30)

momentum_advection = WENOVectorInvariant(vorticity_order=3)
tracer_advection   = Centered()

@info "Defining ocean simulation"

@time ocean = ocean_simulation(grid;
                            momentum_advection,
                            tracer_advection,
                            free_surface)

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(20))
                            
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=1minutes, stop_time=10days)

volmask = CenterField(grid)
set!(volmask, 1)

@info "Defining condition masks"

Atlantic_mask = repeat(basin_mask(grid, "atlantic", volmask), 1, 1, Nz);
IPac_mask = repeat(basin_mask(grid, "indo-pacific", volmask), 1, 1, Nz);
glob_mask = Atlantic_mask .|| IPac_mask;

tracers = ocean.model.tracers
velocities = ocean.model.velocities

#### TRACERS ####

tracer_volmask = [Ax, Δz, volmask]
masks = [glob_mask, Atlantic_mask, IPac_mask]
suffixes = ["_global_", "_atl_", "_ipac_"]
tuples = NamedTuple[]
for j in 1:3
    zonal_int_outputs = ocean_tracer_content(tracers; operator = tracer_volmask[1], dims = (1), condition = masks[j], suffix = suffixes[j]*"zonal")
    depth_int_outputs = ocean_tracer_content(tracers; operator = tracer_volmask[2], dims = (1,2), condition = masks[j], suffix = suffixes[j]*"depth")
    tot_int_outputs = ocean_tracer_content(tracers; operator = tracer_volmask[3], dims = (1,2,3), condition = masks[j], suffix = suffixes[j]*"tot")
    push!(tuples, zonal_int_outputs)
    push!(tuples, depth_int_outputs)
    push!(tuples, tot_int_outputs)
end

@time tracer_outputs = merge(tuples)

#### VELOCITIES ####

transport_volmask_operators = [Ax, Ay, Az]
tuples = NamedTuple[]
for j in range(1:3)
    zonal_int_outputs = volume_transport(velocities; operators = transport_volmask_operators, dims = (1), condition = masks[j], suffix = suffixes[j]*"zonal")
    depth_int_outputs = volume_transport(velocities; operators = transport_volmask_operators, dims = (1,2), condition = masks[j], suffix = suffixes[j]*"depth")
    tot_int_outputs = volume_transport(velocities; operators = transport_volmask_operators, dims = (1,2,3), condition = masks[j], suffix = suffixes[j]*"tot")
    push!(tuples, zonal_int_outputs)
    push!(tuples, depth_int_outputs)
    push!(tuples, tot_int_outputs)
end

@time transport_outputs = merge(tuples)

#=

@time tracer_global_zonal_int_outputs = ocean_tracer_content(tracers; operator = tracer_volmask_zonal, dims = (1), condition = glob_mask, suffix = *"_zonal")
@time tracer_Atlantic_zonal_int_outputs = ocean_tracer_content(tracers; operator = tracer_volmask_zonal, dims = (1), condition = Atlantic_mask, suffix = "_atl_zonal")
@time tracer_IPac_zonal_int_outputs = ocean_tracer_content(tracers; operator = tracer_volmask_zonal, dims = (1), condition = IPac_mask, suffix = "_ipac_zonal")

# Depth Integral
tracer_volmask_depth = Δz
@time tracer_global_depth_int_outputs = ocean_tracer_content(tracers; operator = tracer_volmask_depth, dims = (1,2), condition = glob_mask, suffix = "_global_depth")
@time tracer_Atlantic_depth_int_outputs = ocean_tracer_content(tracers; operator = tracer_volmask_depth, dims = (1,2), condition = Atlantic_mask, suffix = "_atl_depth")
@time tracer_IPac_depth_int_outputs = ocean_tracer_content(tracers; operator = tracer_volmask_depth, dims = (1,2), condition = IPac_mask, suffix = "_ipac_depth")

# Global Integral
tracer_volmask_tot = volmask
@time tracer_global_int_outputs = ocean_tracer_content(tracers; operator = tracer_volmask_tot, dims = (1,2,3), condition = glob_mask, suffix = "_global_tot")
@time tracer_Atlantic_int_outputs = ocean_tracer_content(tracers; operator = tracer_volmask_tot, dims = (1,2,3), condition = Atlantic_mask, suffix = "_atl_tot")
@time tracer_IPac_int_outputs = ocean_tracer_content(tracers; operator = tracer_volmask_tot, dims = (1,2,3), condition = IPac_mask, suffix = "_ipac_tot")

# Zonal Integral
@time transport_global_zonal_int_outputs = volume_transport(velocities; operators = transport_volmask_operators, dims = (1), condition = glob_mask, suffix = "_global_zonal")
@time transport_Atlantic_zonal_int_outputs = volume_transport(velocities; operators = transport_volmask_operators, dims = (1), condition = Atlantic_mask, suffix = "_atl_zonal")
@time transport_IPac_zonal_int_outputs = volume_transport(velocities; operators = transport_volmask_operators, dims = (1), condition = IPac_mask, suffix = "_ipac_zonal")

# Depth Integral
@time tracer_global_depth_int_outputs = volume_transport(velocities; operators = transport_volmask_operators, dims = (1,2), condition = glob_mask, suffix = "_global_depth")
@time transport_Atlantic_depth_int_outputs = volume_transport(velocities; operators = transport_volmask_operators, dims = (1,2), condition = Atlantic_mask, suffix = "_atl_depth")
@time transport_IPac_depth_int_outputs = volume_transport(velocities; operators = transport_volmask_operators, dims = (1,2), condition = IPac_mask, suffix = "_ipac_depth")

# Global Integral
@time transport_global_int_outputs = volume_transport(velocities; operators = transport_volmask_operators, dims = (1,2,3), condition = glob_mask, suffix = "_global_tot")
@time transport_Atlantic_int_outputs = volume_transport(velocities; operators = transport_volmask_operators, dims = (1,2,3), condition = Atlantic_mask, suffix = "_atl_tot")
@time transport_IPac_int_outputs = volume_transport(velocities; operators = transport_volmask_operators, dims = (1,2,3), condition = IPac_mask, suffix = "_ipac_tot")

@time transport_outputs = merge(transport_global_zonal_int_outputs, transport_Atlantic_zonal_int_outputs, transport_IPac_zonal_int_outputs,
                            tracer_global_depth_int_outputs, transport_Atlantic_depth_int_outputs, transport_IPac_depth_int_outputs,
                            transport_global_int_outputs, transport_Atlantic_int_outputs, transport_IPac_int_outputs)

=#

iter_intervals = 2

simulation.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                                 schedule = IterationInterval(iter_intervals),
                                                 filename = "global_surface_fields",
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

simulation.output_writers[:fluxes] = JLD2Writer(coupled_model, fluxes;
                                                schedule = IterationInterval(iter_intervals),
                                                filename = "fluxes",
                                                overwrite_existing = true)

simulation.output_writers[:ocean_tracer_content] = JLD2Writer(ocean.model, tracer_outputs;
                                                          schedule = IterationInterval(iter_intervals),
                                                          filename = "ocean_tracer_content",
                                                          overwrite_existing = true)

simulation.output_writers[:transport] = JLD2Writer(ocean.model, transport_outputs;
                                                          schedule = IterationInterval(iter_intervals),
                                                          filename = "mass_transport",
                                                          overwrite_existing = true)

wall_time = Ref(time_ns())

function progress(sim)
    u, v, w = sim.model.velocities
    T = sim.model.tracers.T
    Tmax = maximum(interior(T))
    Tmin = minimum(interior(T))
    umax = (maximum(abs, interior(u)),
            maximum(abs, interior(v)),
            maximum(abs, interior(w)))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Tmax, Tmin)
    msg4 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg3 * msg4

     wall_time[] = time_ns()

     return nothing
end

add_callback!(simulation, progress, IterationInterval(1))

run!(simulation)
# @btime time_step!($simulation)

# @btime time_step!(simulation.model, 0.1)

# using Oceananigans: write_output!

# writer = simulation.output_writers[:surface]
# @btime write_output!(writer, simulation.model) 

# writer = simulation.output_writers[:zonal_int]
# @btime write_output!(writer, simulation.model) 


# # run!(simulation)
