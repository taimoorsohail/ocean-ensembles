using MPI
using CUDA

MPI.Init()
atexit(MPI.Finalize)  

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Printf
using Dates
using ClimaOcean
using ClimaOcean.EN4
using ClimaOcean.EN4: download_dataset
using ClimaOcean.ECCO
using ClimaOcean.DataWrangling.ETOPO
using JLD2

arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)

localrank = Integer(arch.local_rank)

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

Nx, Ny, Nz = Integer(360/10), Integer(180/10), Integer(50/10)
@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialDiscretization(Nz, depth, 0, mutable=true)

const z_surf = z_faces.cᵃᵃᶠ(Nz)

@info "Creating grid"

underlying_grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (7, 7, 4))

@info "Defining grid"

ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir = data_path)
ClimaOcean.DataWrangling.download_dataset(ETOPOmetadata)

@time bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
                                minimum_depth = 15,
                                interpolation_passes = 1, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
                                major_basins = 6)

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

@info "Creating free surface"
free_surface = SplitExplicitFreeSurface(grid; substeps = 10)

@info "Defining closures"

catke_closure = ClimaOcean.Oceans.default_ocean_closure()  #RiBasedVerticalDiffusivity()#
closure = (catke_closure, VerticalScalarDiffusivity(κ=1e-5, ν=1e-4))
@info "Defining advection schemes"
momentum_advection = WENOVectorInvariant()
tracer_advection   = WENO(order = 7)

dates = vcat(collect(DateTime(1991, 1, 1): Month(1): DateTime(1991, 5, 1)),
             collect(DateTime(1990, 5, 1): Month(1): DateTime(1990, 12, 1)))

@info "We download the 1990-1991 data for an RYF implementation"

dataset = EN4Monthly() # Other options include ECCO2Monthly(), ECCO4Monthly() or ECCO2Daily()

temperature = Metadata(:temperature; dates, dataset = dataset, dir=data_path)
salinity    = Metadata(:salinity;    dates, dataset = dataset, dir=data_path)

download_dataset(temperature)
download_dataset(salinity)

@info "Defining restoring rate"

restoring_rate  = 1 / 30days
@inline mask(x, y, z, t) = z ≥ z_surf - 1

FS = DatasetRestoring(salinity, grid; mask, rate=restoring_rate, time_indices_in_memory = 10)
forcing = (; S=FS)

@info "Defining ocean model"

@time ocean = ocean_simulation(grid; Δt=1minutes,
                         momentum_advection,
                         tracer_advection,
                         timestepper = :SplitRungeKutta3,
                         free_surface,
                         forcing = forcing,
                         radiative_forcing = nothing,
                         closure)

@info "Downloading/checking input data"

set!(ocean.model, T=Metadata(:temperature; dates=first(dates), dataset = dataset, dir=data_path),
                  S=Metadata(:salinity;    dates=first(dates), dataset = dataset, dir=data_path))

#####
##### A Prognostic Sea-ice model
#####

# Default sea-ice dynamics and salinity coupling are included in the defaults
sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7)) 

set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dataset=ECCO4Monthly(), dir=data_path),
                    ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly(), dir=data_path))

# ### Atmospheric forcing

# We force the simulation with an JRA55-do atmospheric reanalysis.
@info "Defining Atmospheric state"

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(100), include_rivers_and_icebergs=true)

# ### Coupled simulation

# Now we are ready to build the coupled ocean--sea ice model and bring everything
# together into a `simulation`.

# We use a relatively short time step initially and only run for a few days to
# avoid numerical instabilities from the initial "shock" of the adjustment of the
# flow fields.

@info "Defining coupled model"
@time coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

@info "Creating simulation"

simulation = Simulation(coupled_model; Δt=60, stop_iteration=12)

# Nice progress messaging is helpful:

## Print a progress message
progress_message(sim) = @printf("Iteration: %04d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.ocean.model.velocities.w), prettytime(sim.run_wall_time))

add_callback!(simulation, progress_message, IterationInterval(1))

if parse(Int,ARGS[2]) == 1 | parse(Int,ARGS[2]) == 7
    tracers = ocean.model.tracers
    velocities = ocean.model.velocities

    outputs = merge(tracers, velocities)

    @info "Saving distributed output writers"
    @time ocean.output_writers[:snapshot] = JLD2Writer(ocean.model, outputs;
                                                dir = output_path,
                                                schedule = AveragedTimeInterval(1minutes),
                                                filename = "test_distributed_outputwriters_outputsonly",
                                                indices = (:, :, Nz),
                                                including = [:grid, :coriolis, :buoyancy, :closure],
                                                with_halos = false,
                                                overwrite_existing = true,
                                                array_type = Array{Float32})
end

if parse(Int,ARGS[2]) == 2 | parse(Int,ARGS[2]) == 7
    @info "Saving distributed surface fields"
    surface_height = (; surface_height = ocean.model.free_surface.displacement)
    surface_forcing = (; T_surf = ocean.model.tracers.T.boundary_conditions.top.condition, 
                        S_surf = ocean.model.tracers.S.boundary_conditions.top.condition)

    outputs_surf = merge(surface_height, surface_forcing)

    @time ocean.output_writers[:SSH] = JLD2Writer(ocean.model, outputs_surf;
                                                dir = output_path,
                                                schedule = AveragedTimeInterval(1minutes),
                                                filename = "test_forcing_fields",
                                                including = [:grid, :coriolis, :buoyancy, :closure],
                                                with_halos = false,
                                                overwrite_existing = true,
                                                array_type = Array{Float32})
end

if parse(Int,ARGS[2]) == 3 | parse(Int,ARGS[2]) == 7

    @info "Saving clock restarts"

    function save_restart(sim)
        localrank = MPI.Comm_rank(MPI.COMM_WORLD)
        jldsave(output_path * "test_clock_checkpoint_rank$(localrank).jld2";
        clock = sim.model.ocean.model.clock)
    end

    add_callback!(simulation, save_restart, TimeInterval(1minutes))
end

if parse(Int,ARGS[2]) == 4 | parse(Int,ARGS[2]) == 7

    @info "Saving ocean checkpointers"

    ocean_checkpointer_tracers = merge(
        ocean.model.velocities,
        ocean.model.tracers,
        ocean.model.free_surface.barotropic_velocities,
        (; η = simulation.model.ocean.model.free_surface.displacement)
    )
    @time ocean.output_writers[:checkpointer_ocean] = JLD2Writer(ocean.model, ocean_checkpointer_tracers;
                                                dir = output_path,
                                                schedule =  TimeInterval(1minutes),
                                                filename = "test_ovar_checkpoint",
                                                with_halos = false,
                                                including = [:grid, :coriolis, :buoyancy, :closure],
                                                overwrite_existing = true)
end

if parse(Int,ARGS[2]) == 5 | parse(Int,ARGS[2]) == 7

    @info "Saving sea-ice checkpointers"

    sea_ice_checkpointer_tracers = merge(  
                                    (h = sea_ice.model.ice_thickness,
                                    ℵ = sea_ice.model.ice_concentration,
                                    Tu = sea_ice.model.ice_thermodynamics.top_surface_temperature,
                                    Gʰ = sea_ice.model.ice_thermodynamics.thermodynamic_tendency),
                                    sea_ice.model.dynamics.auxiliaries.fields, 
                                    sea_ice.model.velocities)


    @time sea_ice.output_writers[:checkpointer_sea_ice] = JLD2Writer(sea_ice.model, sea_ice_checkpointer_tracers;
                                                dir = output_path,
                                                schedule = TimeInterval(1minutes),
                                                filename = "test_sivar_checkpoint_rank$(localrank)",
                                                with_halos = false,
                                                overwrite_existing = true)
end

@info "Running simulation"
run!(simulation)