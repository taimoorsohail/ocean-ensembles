using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
using ClimaOcean.ECCO
using ClimaOcean.ECCO: download_dataset
using OceanEnsembles: basin_mask, ocean_tracer_content!, volume_transport!
using Oceananigans.Operators: Ax, Ay, Az, Δz
using Oceananigans.Fields: ReducedField

## Argument is provided by the submission script!

if isempty(ARGS)
    arch = CPU()
elseif ARGS[2] == "GPU"
    arch = GPU()
elseif ARGS[2] == "CPU"
    arch = CPU()
else
    throw(ArgumentError("Architecture must be provided in the format julia --project example_script.jl --arch GPU --suffix RYF25deg"))
end

@info "Using architecture: ", arch

# ### Download necessary files to run the code

# ### ECCO files
@info "Downloading/checking ECCO data"

dates = vcat(collect(DateTime(1993, 1, 1): Month(1): DateTime(1993, 4, 1)), collect(DateTime(1993, 5, 1) : Month(1) : DateTime(1994, 1, 1)))

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")

temperature = Metadata(:temperature; dates, dataset=ECCO4Monthly(), dir=data_path)
salinity    = Metadata(:salinity;    dates, dataset=ECCO4Monthly(), dir=data_path)

download_dataset(temperature)
download_dataset(salinity)

# ### Grid and Bathymetry
@info "Defining grid"

Nx = Integer(360*4)
Ny = Integer(180*4)
Nz = Integer(100)

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

# ### Restoring
#
# We include temperature and salinity surface restoring to ECCO data.

@info "Defining restoring rate"

restoring_rate  = 1 / 10days
z_below_surface = r_faces[end-1]

mask = LinearlyTaperedPolarMask(southern=(-80, -70), northern=(70, 90), z=(z_below_surface, 0))

FT = ECCORestoring(temperature, grid; mask, rate=restoring_rate)
FS = ECCORestoring(salinity,    grid; mask, rate=restoring_rate)
forcing = (T=FT, S=FS)

# ### Closures
# We include a Gent-McWilliam isopycnal diffusivity as a parameterization for the mesoscale
# eddy fluxes. For vertical mixing at the upper-ocean boundary layer we include the CATKE
# parameterization. We also include some explicit horizontal diffusivity.

@info "Defining closures"

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity,
                                       DiffusiveFormulation

eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3,
                                                 skew_flux_formulation=DiffusiveFormulation())
vertical_mixing = ClimaOcean.OceanSimulations.default_ocean_closure()

closure = (eddy_closure, vertical_mixing)

# ### Ocean simulation
# Now we bring everything together to construct the ocean simulation.
# We use a split-explicit timestepping with 30 substeps for the barotropic
# mode.

@info "Defining free surface"

free_surface = SplitExplicitFreeSurface(grid; substeps=30)

# As per https://github.com/taimoorsohail/ocean-ensembles/issues/38 -- for quarter degree
momentum_advection = WENOVectorInvariant(order=5) 
tracer_advection = WENO(order=5)

@info "Defining ocean simulation"

@time ocean = ocean_simulation(grid;
                         momentum_advection,
                         tracer_advection,
                         free_surface,
                         closure,
                         forcing)

# ### Initial condition

# We initialize the ocean from the ECCO state estimate.

@info "Initialising with ECCO"

set!(ocean.model, T=Metadata(:temperature; dates=first(dates), dataset=ECCO4Monthly()),
                  S=Metadata(:salinity;    dates=first(dates), dataset=ECCO4Monthly()))

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
simulation = Simulation(coupled_model; Δt=30, stop_time=20days)

# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

@info "Defining messenger"

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

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Tmax, Tmin)
    msg4 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg3 * msg4

     wall_time[] = time_ns()

     return nothing
end

# And add it as a callback to the simulation.
add_callback!(simulation, progress, IterationInterval(10))

volmask = CenterField(grid)
set!(volmask, 1)

@info "Defining condition masks"

Atlantic_mask = repeat(basin_mask(grid, "atlantic", volmask), 1, 1, Nz);
IPac_mask = repeat(basin_mask(grid, "indo-pacific", volmask), 1, 1, Nz);
glob_mask = Atlantic_mask .|| IPac_mask;

#### SURFACE

@info "Defining surface outputs"

tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)

#### TRACERS ####

tracer_volmask = [Ax, Δz, volmask]
masks = [glob_mask, Atlantic_mask, IPac_mask]
suffixes = ["_global_", "_atl_", "_ipac_"]
tracer_names = Symbol[]
tracer_outputs = Reduction[]
for j in 1:3
    ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[1], dims = (1), condition = masks[j], suffix = suffixes[j]*"zonal");
    ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[2], dims = (1, 2), condition = masks[j], suffix = suffixes[j]*"depth");
    ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[3], dims = (1, 2, 3), condition = masks[j], suffix = suffixes[j]*"tot");
end

@info "Merging tracer tuples"

tracer_tuple = NamedTuple{Tuple(tracer_names)}(Tuple(tracer_outputs))

#### VELOCITIES ####
@info "Velocities"

transport_volmask_operators = [Ax, Ay, Az]
transport_names = Symbol[]
transport_outputs = ReducedField[]
for j in 1:3
    volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1), condition = masks[j], suffix = suffixes[j]*"zonal")
    volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1,2), condition = masks[j], suffix = suffixes[j]*"depth")
    volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1,2,3), condition = masks[j], suffix = suffixes[j]*"tot")
end

@info "Merging velocity tuples"

transport_tuple = NamedTuple{Tuple(transport_names)}(Tuple(transport_outputs))

constants = simulation.model.interfaces.ocean_properties

@info "Defining output writers"

output_intervals = 5days
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")

simulation.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                                 dir = output_path,
                                                 schedule = TimeInterval(output_intervals),
                                                 filename = "global_surface_fields_$(ARGS[4])",
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

simulation.output_writers[:fluxes] = JLD2Writer(ocean.model, fluxes;
                                                dir = output_path,
                                                schedule = TimeInterval(output_intervals),
                                                filename = "fluxes_$(ARGS[4])",
                                                overwrite_existing = true)

simulation.output_writers[:ocean_tracer_content] = JLD2Writer(ocean.model, tracer_tuple;
                                                          dir = output_path,
                                                          schedule = TimeInterval(output_intervals),
                                                          filename = "ocean_tracer_content_$(ARGS[4])",
                                                          overwrite_existing = true)

simulation.output_writers[:transport] = JLD2Writer(ocean.model, transport_tuple;
                                                          dir = output_path,
                                                          schedule = TimeInterval(output_intervals),
                                                          filename = "mass_transport_$(ARGS[4])",
                                                          overwrite_existing = true)

@info "Running simulation"

run!(simulation)

simulation.Δt = 5minutes
simulation.stop_time = 11000days

run!(simulation)