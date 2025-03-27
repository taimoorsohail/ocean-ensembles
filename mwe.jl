using Oceananigans
using Oceananigans.Units
using BenchmarkTools
includet("BasinMask.jl")
using .BasinMask

Nx = Integer(360/4)
Ny = Integer(180/4)
Nz = Integer(100/4)

arch = CPU()
z_faces = (-4000, 0)
# grid = LatitudeLongitudeGrid(size=(60, 50, Nz), longitude=(0, 360), latitude=(-75, 75), z=(-2000, 0))
grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (5, 5, 4),
                    first_pole_longitude = 70,
                    north_poles_latitude = 55)

free_surface = SplitExplicitFreeSurface(grid; substeps=10)
model = HydrostaticFreeSurfaceModel(; grid, free_surface)

# set!(model, u=0.001*rand(), v=0.001*rand())

simulation = Simulation(model, Δt=0.001, stop_iteration=2000)
pop!(simulation.callbacks, :nan_checker)


function integrate_tuple(outputs; volmask, dims, condition, suffix::AbstractString) # Add suffix kwarg
    int_model_outputs = NamedTuple((Symbol(string(key) * suffix) => Integral(outputs[key]; dims, condition) for key in keys(outputs)))
    dV_int = NamedTuple{(Symbol(:dV, suffix),)}((Integral(volmask; dims, condition),))
    int_outputs = merge(int_model_outputs, dV_int)
    return int_outputs
end

c = CenterField(grid)
volmask =  set!(c, 1)

@info "Defining masks"

Atlantic_mask = repeat(basin_mask(grid, "atlantic", c, arch), 1, 1, Nz)
IPac_mask = repeat(basin_mask(grid, "indo-pacific", c, arch), 1, 1, Nz)
glob_mask = Atlantic_mask .|| IPac_mask

# @info "Defining surface outputs"

tracers = model.tracers
velocities = model.velocities

outputs = merge(tracers, velocities)

@time global_zonal_int_outputs = integrate_tuple(outputs; volmask, dims = (1), condition = glob_mask, suffix = "_global")
@time Atlantic_zonal_int_outputs = integrate_tuple(outputs; volmask, dims = (1), condition = Atlantic_mask, suffix = "_atlantic")
@time IPac_zonal_int_outputs = integrate_tuple(outputs; volmask, dims = (1), condition = IPac_mask, suffix = "_pacific")

@time zonal_int_outputs = merge(global_zonal_int_outputs, Atlantic_zonal_int_outputs, IPac_zonal_int_outputs)

simulation.output_writers[:surface] = JLD2Writer(model, outputs;
                                                 schedule = IterationInterval(10),
                                                 filename = "global_surface_fields",
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

simulation.output_writers[:zonal_int] = JLD2Writer(model, (u = zonal_int_outputs[1], v = zonal_int_outputs[2]);
                                                          schedule = IterationInterval(10),
                                                          filename = "zonally_integrated_data",
                                                          overwrite_existing = true)


using Printf
using Oceananigans: write_output!

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

# run!(simulation)
# @btime time_step!($simulation)

# @btime time_step!(simulation.model, 0.1)

# using Oceananigans: write_output!

# writer = simulation.output_writers[:surface]
# @btime write_output!(writer, simulation.model) 

# writer = simulation.output_writers[:zonal_int]
# @btime write_output!(writer, simulation.model) 


# # run!(simulation)
