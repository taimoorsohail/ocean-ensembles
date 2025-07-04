using ClimaOcean
using Oceananigans
using Oceananigans.Units
import Oceananigans.Architectures: on_architecture
using CFTime
using Dates
using Printf
using OceanEnsembles
using Oceananigans.Operators: Ax, Ay, Az, Δz
using Oceananigans.Fields: ReducedField
using ClimaOcean.EN4
using ClimaOcean.EN4: download_dataset
using ClimaOcean.ECCO
using ClimaOcean.ECCO: download_dataset
using JLD2 

# using ClimaOcean.DataWrangling: Restoring

# ### EN4 files
@info "Downloading/checking data"
## We download Gouretski and Reseghetti (2010) XBT corrections and Gouretski and Cheng (2020) MBT corrections

dates = collect(DateTime(1993, 1, 1): Month(1): DateTime(1994, 1, 1))

data_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles/data/")

dataset = EN4Monthly() # Other options include ECCO2Monthly(), ECCO4Monthly() or ECCO2Daily()

temperature = Metadata(:temperature; dates, dataset = dataset, dir=data_path)
salinity    = Metadata(:salinity;    dates, dataset = dataset, dir=data_path)

download_dataset(temperature)
download_dataset(salinity)

Nx = Integer(360)
Ny = Integer(180)
Nz = Integer(100/2)

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
# view(bottom_height, 102:103, 124, 1) .= -400

@info "Defining grid"

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

@info "Defining restoring rate"

restoring_rate  = 2 / 365.25days

#mask = LinearlyTaperedPolarMask(southern=(-90, 0), northern=(0, 90), z=(z_below_surface, 0))

FT = DatasetRestoring(temperature, grid; rate=restoring_rate)
FS = DatasetRestoring(salinity,    grid; rate=restoring_rate)
forcing = (T=FT, S=FS)

@info "Defining free surface"

free_surface = SplitExplicitFreeSurface(grid; substeps=50)

momentum_advection = WENOVectorInvariant(vorticity_order=5)
tracer_advection   = Centered()

@info "Defining ocean simulation"

@time ocean = ocean_simulation(grid; free_surface,
                                momentum_advection,
                                tracer_advection)

@info "Initialising with EN4"

set!(ocean.model, T=Metadata(:temperature; dates=first(dates), dataset = dataset, dir=data_path),
                    S=Metadata(:salinity;    dates=first(dates), dataset = dataset, dir=data_path))

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(20))

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=2minutes, stop_time=10days)

tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)

@info "Defining output tuples"
@info "Tracers"

#### TRACERS ####

volmask = CenterField(grid)
set!(volmask, 1)
wmask = ZFaceField(grid)

@info "Defining condition masks"

Atlantic_mask = basin_mask(grid, "atlantic", volmask);
IPac_mask = basin_mask(grid, "indo-pacific", volmask);
glob_mask = Atlantic_mask .|| IPac_mask;

tracer_volmask = [Ax, Δz, volmask]
masks_centers = [repeat(glob_mask, 1, 1, size(volmask)[3]),
         repeat(Atlantic_mask, 1, 1, size(volmask)[3]),
         repeat(IPac_mask, 1, 1, size(volmask)[3])]
masks_wfaces = [repeat(glob_mask, 1, 1, size(wmask)[3]),
         repeat(Atlantic_mask, 1, 1, size(wmask)[3]),
         repeat(IPac_mask, 1, 1, size(wmask)[3])]

masks = [
            [masks_centers[1], masks_wfaces[1]],  # Global
            [masks_centers[2], masks_wfaces[2]],  # Atlantic
            [masks_centers[3], masks_wfaces[3]]   # IPac
        ]

suffixes = ["_global_", "_atl_", "_ipac_"]
tracer_names = Symbol[]
tracer_outputs = Reduction[]

for j in 1:3
    @time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[1], dims = (1), condition = masks[j][1], suffix = suffixes[j]*"zonal");
    @time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[2], dims = (1, 2), condition = masks[j][1], suffix = suffixes[j]*"depth");
    @time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[3], dims = (1, 2, 3), condition = masks[j][1], suffix = suffixes[j]*"tot");
end

@info "Merging tracer tuples"

tracer_tuple = NamedTuple{Tuple(tracer_names)}(Tuple(tracer_outputs))

#### VELOCITIES ####
@info "Velocities"

transport_volmask_operators = [Ax, Ay, Az]
transport_names = Symbol[]
transport_outputs = ReducedField[]
for j in 1:3
    @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1), condition = masks[j], suffix = suffixes[j]*"zonal")
    @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1,2), condition = masks[j], suffix = suffixes[j]*"depth")
    @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1,2,3), condition = masks[j], suffix = suffixes[j]*"tot")
end

@info "Merging velocity tuples"

transport_tuple = NamedTuple{Tuple(transport_names)}(Tuple(transport_outputs))

output_intervals =  AveragedTimeInterval(5days)
callback_interval = IterationInterval(1)

output_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles/outputs/")

simulation.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                                 dir = output_path,
                                                 schedule = output_intervals,
                                                 filename = "global_surface_fields",
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

simulation.output_writers[:fluxes] = JLD2Writer(ocean.model, fluxes;
                                                dir = output_path,
                                                schedule = output_intervals,
                                                filename = "fluxes",
                                                overwrite_existing = true)

simulation.output_writers[:ocean_tracer_content] = JLD2Writer(ocean.model, tracer_tuple;
                                                          dir = output_path,
                                                          schedule = output_intervals,
                                                          filename = "ocean_tracer_content",
                                                          overwrite_existing = true)

simulation.output_writers[:transport] = JLD2Writer(ocean.model, transport_tuple;
                                                          dir = output_path,
                                                          schedule = output_intervals,
                                                          filename = "mass_transport",
                                                          overwrite_existing = true)

wall_time = Ref(time_ns())

function save_nans(sim)
    if simulation.callbacks[:nan_checker].func.erroring == true
        ucpu = on_architecture(CPU(), sim.model.ocean.model.velocities.u)
        vcpu = on_architecture(CPU(), sim.model.ocean.model.velocities.v)
        Tcpu = on_architecture(CPU(), sim.model.ocean.model.tracers.T)
        Scpu = on_architecture(CPU(), sim.model.ocean.model.tracers.S)

        JLD2.@save output_path * "nan_state.jld2" ucpu vcpu Tcpu Scpu
        throw(ErrorException("NaNs detected - Saved field."))
    end 
end

function progress(sim)

    u, v, w = sim.model.ocean.model.velocities
    T, S, e = sim.model.ocean.model.tracers
    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    erange = (maximum((e)), minimum((e)))

    umax = (maximum(abs, (u)),
            maximum(abs, (v)),
            maximum(abs, (w)))
    
    save_nans(sim)

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

run!(simulation)

simulation.Δt = 20minutes
simulation.stop_time = 11000days

run!(simulation)
