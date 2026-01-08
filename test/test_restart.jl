using MPI
using CUDA

MPI.Init()
atexit(MPI.Finalize)  

using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
using Oceananigans.DistributedComputations
using OceanEnsembles
using JLD2
using Test
using Glob
using Oceananigans.Architectures: on_architecture

# File paths
data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")
target_time = 365days
## Argument is provided by the submission script!

arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
total_ranks = MPI.Comm_size(MPI.COMM_WORLD)

@info "Using architecture: " * string(arch)

restartfiles = glob("checkpoint_iteration*", output_path)

# Extract the numeric suffix from each filename
restart_numbers = map(f -> parse(Int, match(r"checkpoint_iteration(\d+)", basename(f)).captures[1]), restartfiles)

iteration = 0
time = 0.0
if !isempty(restart_numbers) && maximum(restart_numbers) != 0
    # Extract the numeric suffix from each filename

    # Get the file with the maximum number
    clock_vars = jldopen(output_path * "checkpoint_iteration" * string(maximum(restart_numbers)) * "_rank" * string(arch.local_rank) * ".jld2")

    iteration = deepcopy(clock_vars["clock"].iteration)
    time = deepcopy(clock_vars["clock"].time)
    @info "Moving simulation to " * string(iteration) * " iterations"
    @info "Moving simulation to " * string(prettytime(time))

    close(clock_vars)
end

if time == target_time
    error("Terminating simulation at target time.")
end

# ### Grid and Bathymetry
@info "Defining grid"

Nx = Integer(36*3)
Ny = Integer(18*3)
Nz = Integer(40)

@info "Defining vertical z faces"

r_faces = exponential_z_faces(; Nz, depth=5000, h=12.43)

@info "Defining tripolar grid"

underlying_grid = TripolarGrid(arch;
                               size = (Nx, Ny, Nz),
                               z = r_faces,
                               halo = (5, 5, 4),
                               first_pole_longitude = 70,
                               north_poles_latitude = 55)

@info "Done defining tripolar grid"

@info "Defining bottom bathymetry"

@time bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 1, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
				                  major_basins = 2)

# For this bathymetry at this horizontal resolution we need to manually open the Gibraltar strait.
# view(bottom_height, 102:103, 124, 1) .= -400

@info "Defining grid"

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

@info "Defining closures"

eddy_closure = Oceananigans.TurbulenceClosures.IsopycnalSkewSymmetricDiffusivity(κ_skew=2e3, κ_symmetric=2e3)
vertical_mixing = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity(minimum_tke=1e-6)
horizontal_viscosity = HorizontalScalarDiffusivity(ν=4000)
closure = (eddy_closure, horizontal_viscosity, vertical_mixing)

# ### Ocean simulation
# Now we bring everything together to construct the ocean simulation.
# We use a split-explicit timestepping with 30 substeps for the barotropic
# mode.

@info "Defining free surface"

free_surface       = SplitExplicitFreeSurface(grid; substeps=70)
momentum_advection = VectorInvariant()
tracer_advection   = WENO(order=5)

@info "Defining ocean simulation"

@time ocean = ocean_simulation(grid;
                         momentum_advection,
                         tracer_advection,
                         free_surface,
                         closure)

# We force the simulation with an JRA55-do atmospheric reanalysis.
@info "Defining Atmospheric state"

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(20))

# ### Coupled simulation

# Now we are ready to build the coupled ocean--sea ice model and bring everything
# together into a `simulation`.

# We use a relatively short time step initially and only run for a few days to
# avoid numerical instabilities from the initial "shock" of the adjustment of the
# flow fields.

@info "Defining coupled model"

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=5minutes, stop_time=10days)

time = 0.0

@show prettytime(time)
@show iteration

simulation.model.ocean.model.clock.iteration = iteration
simulation.model.ocean.model.clock.time = time
simulation.model.atmosphere.clock.iteration = iteration
simulation.model.atmosphere.clock.time = time
simulation.model.clock.iteration = iteration
simulation.model.clock.time = time

# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = IterationInterval(20)

function progress(sim)
    u, v, w = sim.model.ocean.model.velocities
    T, S, e = sim.model.ocean.model.tracers
    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    erange = (maximum((e)), minimum((e)))

    umax = (maximum(abs, (u)),
            maximum(abs, (v)),
            maximum(abs, (w)))
        
    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Trange...)
    msg4 = @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Srange...)
    msg5 = @sprintf("extrema(e): (%.2f, %.2f) J, ", erange...)
    msg6 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg3 * msg4 * msg5 * msg6

    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, callback_interval)

#### SURFACE

@info "Defining surface outputs"
tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)

output_intervals = AveragedTimeInterval(5days)
checkpoint_intervals = TimeInterval(5days)

simulation.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                                 dir = output_path,
                                                 schedule = output_intervals,
                                                 filename = "global_surface_fields_distributedGPU_iteration" * string(simulation.model.clock.iteration),
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

simulation.output_writers[:fluxes] = JLD2Writer(ocean.model, fluxes;
                                                dir = output_path,
                                                schedule = output_intervals,
                                                filename = "fluxes_distributedGPU_iteration" * string(simulation.model.clock.iteration),
                                                overwrite_existing = true)

# simulation.output_writers[:full_field] = JLD2Writer(ocean.model, outputs;
#                                                 dir = output_path,
#                                                 schedule = restart_interval,
#                                                 filename = restart_file * "_" * string(simulation.model.clock.iteration),
#                                                 with_halos = false,
#                                                 overwrite_existing = true,
#                                                 file_splitting = restart_interval
#                                                 array_type = Array{Float32})

#                                                 @info "Saving restart"

# simulation.output_writers[:checkpoint] = Checkpointer(ocean.model;
#                                         dir = output_path,
#                                         schedule = checkpoint_intervals,
#                                         prefix = "test_checkpoint",
#                                         properties = [],
#                                         overwrite_existing = true)
function save_restart(sim)
    @info @sprintf("Saving checkpoint file")

    jldsave(output_path * "checkpoint_iteration" * string(sim.model.clock.iteration) * "_rank" * string(arch.local_rank) * ".jld2";
    u = on_architecture(CPU(), interior(sim.model.ocean.model.velocities.u)),
    v = on_architecture(CPU(), interior(sim.model.ocean.model.velocities.v)),
    w = on_architecture(CPU(), interior(sim.model.ocean.model.velocities.w)),
    T = on_architecture(CPU(), interior(sim.model.ocean.model.tracers.T)),
    S = on_architecture(CPU(), interior(sim.model.ocean.model.tracers.S)),
    e = on_architecture(CPU(), interior(sim.model.ocean.model.tracers.e)),
    clock = sim.model.clock)

    # # Delete older checkpoint files for this rank
    # files = readdir(output_path; join=true)
    # for file in files
    #     if occursin("checkpoint_iteration", file) &&
    #         occursin("_rank$(arch.local_rank).jld2", file)

    #         # Extract the iteration number using a regex
    #         m = match(r"checkpoint_iteration(\d+)_rank$arch.local_rank\.jld2", basename(file))
    #         if m !== nothing
    #             old_iter = parse(Int, m.captures[1])
    #             if old_iter < sim.model.clock.iteration
    #                 rm(file; force=true)
    #             end
    #         end
    #     end
    # end
end
                                                                                                        
add_callback!(simulation, save_restart, checkpoint_intervals)

if !isempty(restart_numbers) && maximum(restart_numbers) != 0
    @info "Loading with restart file from iteration " * string(maximum(restart_numbers))
    @info "Local rank ", arch.local_rank
    @info output_path * "checkpoint_iteration" * string(maximum(restart_numbers)) * "_rank" * string(arch.local_rank) * ".jld2"

    fields = jldopen(output_path * "checkpoint_iteration" * string(maximum(restart_numbers)) * "_rank" * string(arch.local_rank) * ".jld2")

    T_field = fields["T"]
    S_field = fields["S"]
    e_field = fields["e"]
    u_field = fields["u"]
    v_field = fields["v"]
    w_field = fields["w"]

    close(fields)

    set!(ocean.model, 
    T = (T_field),
    S = (S_field),
    u = (u_field),
    v = (v_field),
    w = (w_field),
    e = (e_field))

    u, v, w = ocean.model.velocities
    T, S, e = ocean.model.tracers

    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    erange = (maximum((e)), minimum((e)))

    umax = (maximum(abs, (u)),
            maximum(abs, (v)),
            maximum(abs, (w)))

    @info Trange, Srange, erange, umax
    
    @info "Restart found at " * string(prettytime(time))

    simulation.Δt = 5minutes 
    simulation.stop_time = target_time # 1 year 

    run!(simulation)
else
    @info "Running simulation"

    run!(simulation)

    # simulation.Δt = 5minutes 
    # simulation.stop_time = target_time # 1 year 

    # run!(simulation)

    # combine_outputs(total_ranks, output_path * "restart_distributedGPU_" * string(simulation.model.clock.iteration),
    #  output_path * "restart_distributedGPU_" * string(simulation.model.clock.iteration); remove_split_files = true)
    # combine_outputs(total_ranks, output_path * "fluxes_distributedGPU_" * string(simulation.model.clock.iteration),
    #  output_path * "fluxes_distributedGPU_" * string(simulation.model.clock.iteration); remove_split_files = true)
    # combine_outputs(total_ranks, output_path * "global_surface_fields_distributedGPU_" * string(simulation.model.clock.iteration),
    #  output_path * "global_surface_fields_distributedGPU_" * string(simulation.model.clock.iteration); remove_split_files = true)
    # combine_outputs(total_ranks, output_path * "clock_distributedGPU_" * string(simulation.model.clock.iteration),
    #  output_path * "clock_distributedGPU_" * string(simulation.model.clock.iteration); remove_split_files = true)

    #  @test isfile(output_path * "restart_distributedGPU_" * string(simulation.model.clock.iteration) * ".jld2")
    #  @test isfile(output_path * "fluxes_distributedGPU_" * string(simulation.model.clock.iteration) * ".jld2")
    #  @test isfile(output_path * "global_surface_fields_distributedGPU_" * string(simulation.model.clock.iteration) * ".jld2")
    #  @test isfile(output_path * "clock_distributedGPU_" * string(simulation.model.clock.iteration) * ".jld2")
end
                                            
