using MPI
using CUDA

MPI.Init()


using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf

# File paths
data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")

## Argument is provided by the submission script!

arch = Distributed(GPU())

@info "Using architecture: " * string(arch)

# ### Grid and Bathymetry
@info "Defining grid"

Nx = Integer(360/5)
Ny = Integer(180/5)
Nz = Integer(50/2)

@info "Defining vertical z faces"

r_faces = exponential_z_faces(; Nz, depth=5000, h=34)
z_faces = Oceananigans.MutableVerticalDiscretization(r_faces)

@info "Defining tripolar grid"

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

# ### Closures
# We include a Gent-McWilliam isopycnal diffusivity as a parameterization for the mesoscale
# eddy fluxes. For vertical mixing at the upper-ocean boundary layer we include the CATKE
# parameterization. We also include some explicit horizontal diffusivity.

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
                         closure,
                         forcing)

# ### Initial condition

# ### Atmospheric forcing

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
simulation = Simulation(coupled_model; Δt=5minutes, stop_time=20days)

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
output_intervals = AveragedTimeInterval(5days)

@info "Defining output writers"

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

@info "Running simulation"

run!(simulation)

# simulation.Δt = 20minutes
# simulation.stop_time = 1826.25days # 5 years

# run!(simulation)

MPI.Finalize()
