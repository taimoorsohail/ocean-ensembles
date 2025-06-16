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

# File paths
data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

## Argument is provided by the submission script!

arch = Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
total_ranks = MPI.Comm_size(MPI.COMM_WORLD)

@info "Using architecture: " * string(arch)

# clockfiles = glob("clock_distributedGPU*", output_path)

# iteration = 0
# time = 0.0
# if !isempty(clockfiles)
#     # Extract the numeric suffix from each filename
#     numbers = map(f -> parse(Int, match(r"clock_distributedGPU_(\d+)", basename(f)).captures[1]), clockfiles)

#     # Get the file with the maximum number
#     max_index = argmax(numbers)
#     maxiter = clockfiles[max_index]

#     clock_vars = jldopen(maxiter)

#     iteration = clock_vars["iteration"]
#     time = clock_vars["time"]
#     @info "Moving simulation to " * string(iteration) * " iterations"
#     @info "Moving simulation to " * string(time) * " time"

# end

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

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), iteration(sim), prettytime(sim.Δt))
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

@info "Saving restart"

output_intervals = AveragedTimeInterval(5days)
checkpoint_intervals = TimeInterval(5days)

simulation.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                                 dir = output_path,
                                                 schedule = output_intervals,
                                                 filename = "global_surface_fields_distributedGPU",
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

simulation.output_writers[:fluxes] = JLD2Writer(ocean.model, fluxes;
                                                dir = output_path,
                                                schedule = output_intervals,
                                                filename = "fluxes_distributedGPU",
                                                overwrite_existing = true)

simulation.output_writers[:checkpoint] = Checkpointer(ocean.model;
                                        dir = output_path,
                                        schedule = checkpoint_intervals,
                                        prefix = "test_checkpoint",
                                        properties = [:grid, :particles, :clock],
                                        overwrite_existing = true)
# function save_clock(sim)
#     if MPI.Comm_rank(MPI.COMM_WORLD) == 0
#         jldsave(output_path * "clock_distributedGPU_" * string(sim.model.clock.iteration) * ".jld2",
#                 sim.model.clock)
#     end
# end
                                                                                                        
# add_callback!(simulation, save_clock, TimeInterval(5days))

restartfiles = glob("test_checkpoint*", output_path)

if !isempty(restartfiles)
    # Extract the numeric suffix from each filename
    numbers = map(f -> parse(Int, match(r"test_checkpoint_(\d+)", basename(f)).captures[1]), restartfiles)

    # Get the file with the maximum number
    max_index = argmax(numbers)
    maxiter = restartfiles[max_index]

    # @info "Loading with restart file from iteration" * string(max(numbers))

    # T_field = FieldTimeSeries(maxiter, "T")
    # S_field = FieldTimeSeries(maxiter, "S")
    # u_field = FieldTimeSeries(maxiter, "u")
    # v_field = FieldTimeSeries(maxiter, "v")
    # w_field = FieldTimeSeries(maxiter, "w")
    # e_field = FieldTimeSeries(maxiter, "e")

    # set!(ocean.model, 
    # T=T_field,
    # S=S_field,
    # u=u_field,
    # v=v_field,
    # w=w_field,
    # e=e_field)

    # Trange = (maximum((T)), minimum((T)))
    # Srange = (maximum((S)), minimum((S)))
    # erange = (maximum((e)), minimum((e)))

    # umax = (maximum(abs, (u)),
    #         maximum(abs, (v)),
    #         maximum(abs, (w)))

    # @test abs(Trange[1])>0
    # @test abs(Trange[2])>0
    # @test abs(Srange[1])>0
    # @test abs(Srange[2])>0
    # @test abs(erange[1])>0
    # @test abs(erange[2])>0
    # @test abs(umax[1])>0
    # @test abs(umax[2])>0
    # @test abs(umax[3])>0

    @info "Running simulation"

    run!(simulation, pickup = maxiter)

    # combine_outputs(total_ranks, output_path * "restart_distributedGPU_" * string(simulation.model.clock.iteration),
    #  output_path * "restart_distributedGPU_" * string(simulation.model.clock.iteration); remove_split_files = true)
    # combine_outputs(total_ranks, output_path * "clock_distributedGPU_" * string(simulation.model.clock.iteration),
    #  output_path * "clock_distributedGPU_" * string(simulation.model.clock.iteration); remove_split_files = true)
#     combine_outputs(total_ranks, output_path * "fluxes_distributedGPU_" * string(simulation.model.clock.iteration),
#     output_path * "fluxes_distributedGPU_" * string(simulation.model.clock.iteration); remove_split_files = true)
#    combine_outputs(total_ranks, output_path * "global_surface_fields_distributedGPU_" * string(simulation.model.clock.iteration),
#     output_path * "global_surface_fields_distributedGPU_" * string(simulation.model.clock.iteration); remove_split_files = true)


    #  @test isfile(output_path * "restart_distributedGPU_" * string(simulation.model.clock.iteration) * ".jld2")
    #  @test isfile(output_path * "fluxes_distributedGPU_" * string(simulation.model.clock.iteration) * ".jld2")
    #  @test isfile(output_path * "global_surface_fields_distributedGPU_" * string(simulation.model.clock.iteration) * ".jld2")
    #  @test isfile(output_path * "clock_distributedGPU_" * string(simulation.model.clock.iteration) * ".jld2")

else
    @info "Running simulation"

    run!(simulation)

    simulation.Δt = 10minutes 
    simulation.stop_time = 365days # 1 year 

    run!(simulation)

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
                                            