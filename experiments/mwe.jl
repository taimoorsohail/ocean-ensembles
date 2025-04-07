using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
using OceanEnsembles: basin_mask, integrate_tracer, integrate_transport
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

# function integrate_tuple(outputs; volmask, dims, condition, suffix::AbstractString) # Add suffix kwarg
#     int_model_outputs = NamedTuple((Symbol(string(key) * suffix) => Integral(outputs[key]; dims, condition) for key in keys(outputs)))
#     dV_int = NamedTuple{(Symbol(:dV, suffix),)}((Integral(volmask; dims, condition),))
#     int_outputs = merge(int_model_outputs, dV_int)
#     return int_outputs
# end

volmask = CenterField(grid)
set!(volmask, 1)

@info "Defining condition masks"

Atlantic_mask = repeat(basin_mask(grid, "atlantic", volmask), 1, 1, Nz)
IPac_mask = repeat(basin_mask(grid, "indo-pacific", volmask), 1, 1, Nz)
glob_mask = Atlantic_mask .|| IPac_mask

# @info "Defining surface outputs"

tracers = ocean.model.tracers
velocities = ocean.model.velocities

# Zonal Integral
tracer_volmask_zonal = Ax
@time global_zonal_int_outputs = integrate_tracer(tracers; metric = tracer_volmask_zonal, dims = (1), condition = glob_mask, suffix = "_global_zonal")
@time Atlantic_zonal_int_outputs = integrate_tracer(tracers; metric = tracer_volmask_zonal, dims = (1), condition = Atlantic_mask, suffix = "_atl_zonal")
@time IPac_zonal_int_outputs = integrate_tracer(tracers; metric = tracer_volmask_zonal, dims = (1), condition = IPac_mask, suffix = "_ipac_zonal")

# Depth Integral
tracer_volmask_depth = Δz
@time global_depth_int_outputs = integrate_tracer(tracers; tracer_volmask_depth, dims = (1,2), condition = glob_mask, suffix = "_global_depth")
@time Atlantic_depth_int_outputs = integrate_tracer(tracers; tracer_volmask_depth, dims = (1,2), condition = Atlantic_mask, suffix = "_atl_depth")
@time IPac_depth_int_outputs = integrate_tracer(tracers; tracer_volmask_depth, dims = (1,2), condition = IPac_mask, suffix = "_ipac_depth")

# Global Integral
tracer_volmask_tot = volmask
@time tracer_global_int_outputs = integrate_tracer(tracers; tracer_volmask_tot, dims = (1,2,3), condition = glob_mask, suffix = "_global_tot")
@time tracer_Atlantic_int_outputs = integrate_tracer(tracers; tracer_volmask_tot, dims = (1,2,3), condition = Atlantic_mask, suffix = "_atl_tot")
@time tracer_IPac_int_outputs = integrate_tracer(tracers; tracer_volmask_tot, dims = (1,2,3), condition = IPac_mask, suffix = "_ipac_tot")

@time tracer_outputs = merge(global_zonal_int_outputs, Atlantic_zonal_int_outputs, IPac_zonal_int_outputs,
                            global_depth_int_outputs, Atlantic_depth_int_outputs, IPac_depth_int_output,
                            global_global_int_outputs, Atlantic_global_int_outputs, IPac_global_int_outputs)

transport_volmask_operators = [Ax, Ay, Az]

@time velocity_global_int_outputs = integrate_transport(first(velocities); metrics = transport_volmask_operators, dims = (1), condition = glob_mask, suffix = "_global_zonal")

# simulation.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
#                                                  schedule = IterationInterval(10),
#                                                  filename = "global_surface_fields",
#                                                  indices = (:, :, grid.Nz),
#                                                  with_halos = false,
#                                                  overwrite_existing = true,
#                                                  array_type = Array{Float32})

fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

simulation.output_writers[:fluxes] = JLD2Writer(coupled_model, fluxes;
                                                schedule = IterationInterval(10),
                                                filename = "fluxes",
                                                overwrite_existing = true)

# simulation.output_writers[:zonal_int] = JLD2Writer(ocean.model, zonal_int_outputs;
#                                                           schedule = IterationInterval(10),
#                                                           filename = "zonally_integrated_data",
#                                                           overwrite_existing = true)

wall_time = Ref(time_ns())

function progress(sim)
    u, v, w = sim.model.velocities
    # T = sim.model.tracers.T
    # Tmax = maximum(interior(T))
    # Tmin = minimum(interior(T))
    umax = (maximum(abs, interior(u)),
            maximum(abs, interior(v)),
            maximum(abs, interior(w)))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    # msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Tmax, Tmin)
    msg4 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg4

     wall_time[] = time_ns()

     return nothing
end


function memory_allocations(sim)

    writer = sim.output_writers[:zonal_int]

    results = @benchmark write_output!(writer, simulation.model)    

    msg1 = @sprintf("memory: %s kb, allocations: %d", results.memory / KiB, results.allocs)
    
    @info msg1

     return nothing
end

add_callback!(simulation, progress, IterationInterval(5))
add_callback!(simulation, memory_allocations, IterationInterval(50))

run!(simulation)
# @btime time_step!($simulation)

# @btime time_step!(simulation.model, 0.1)

# using Oceananigans: write_output!

# writer = simulation.output_writers[:surface]
# @btime write_output!(writer, simulation.model) 

# writer = simulation.output_writers[:zonal_int]
# @btime write_output!(writer, simulation.model) 


# # run!(simulation)
