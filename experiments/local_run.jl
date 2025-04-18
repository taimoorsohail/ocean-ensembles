using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
using OceanEnsembles
using Oceananigans.Operators: Ax, Ay, Az, Δz
using Oceananigans.Fields: ReducedField
using ClimaOcean.ECCO
using ClimaOcean.ECCO: download_dataset


# ### ECCO files
@info "Downloading/checking ECCO data"

dates = vcat(collect(DateTime(1993, 1, 1): Month(1): DateTime(1993, 4, 1)), collect(DateTime(1993, 5, 1) : Month(1) : DateTime(1994, 1, 1)))

data_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/data/")

temperature = Metadata(:temperature; dates, dataset=ECCO4Monthly(), dir=data_path)
salinity    = Metadata(:salinity;    dates, dataset=ECCO4Monthly(), dir=data_path)

download_dataset(temperature)
download_dataset(salinity)

Nx = Integer(360/5)
Ny = Integer(180/5)
Nz = Integer(100/4)

arch = CPU()

z_faces = (-4000, 0)



# underlying_grid = TripolarGrid(arch;
#                                size = (Nx, Ny, Nz),
#                                z = z_faces,
#                                halo = (5, 5, 4),
#                                first_pole_longitude = 70,
#                                north_poles_latitude = 55)

underlying_grid = LatitudeLongitudeGrid(arch;
                                        size = (Nx, Ny, Nz),
                                        z = z_faces,
                                        halo = (5, 5, 4),
                                        longitude = (0, 360),
                                        latitude = (-75, 75))

@info "Defining bottom bathymetry"

@time bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 75, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
				  major_basins = 2)

# For this bathymetry at this horizontal resolution we need to manually open the Gibraltar strait.
# view(bottom_height, 102:103, 124, 1) .= -400

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

@info "Initialising with ECCO"

set!(ocean.model, T=Metadata(:temperature; dates=first(dates), dataset=ECCO4Monthly()),
                    S=Metadata(:salinity;    dates=first(dates), dataset=ECCO4Monthly()))

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(20))

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=2minutes, stop_time=10days)

# volmask = CenterField(grid)
# set!(volmask, 1)

# @info "Defining condition masks"

# Atlantic_mask = basin_mask(grid, "atlantic", volmask);
# IPac_mask = basin_mask(grid, "indo-pacific", volmask);
# glob_mask = Atlantic_mask .|| IPac_mask;

tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)

# @info "Defining output tuples"
# @info "Tracers"

# #### TRACERS ####

# tracer_volmask = [Ax, Δz, volmask]
# masks = [glob_mask, Atlantic_mask, IPac_mask]
# suffixes = ["_global_", "_atl_", "_ipac_"]
# tracer_names = Symbol[]
# tracer_outputs = Reduction[]
# for j in 1:3
#     @time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[1], dims = (1), condition = masks[j], suffix = suffixes[j]*"zonal");
#     @time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[2], dims = (1, 2), condition = masks[j], suffix = suffixes[j]*"depth");
#     @time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[3], dims = (1, 2, 3), condition = masks[j], suffix = suffixes[j]*"tot");
# end

# @info "Merging tracer tuples"

# tracer_tuple = NamedTuple{Tuple(tracer_names)}(Tuple(tracer_outputs))

# #### VELOCITIES ####
# @info "Velocities"

# transport_volmask_operators = [Ax, Ay, Az]
# transport_names = Symbol[]
# transport_outputs = ReducedField[]
# for j in 1:3
#     @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1), condition = masks[j], suffix = suffixes[j]*"zonal")
#     @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1,2), condition = masks[j], suffix = suffixes[j]*"depth")
#     @time volume_transport!(transport_names, transport_outputs; outputs = velocities, operators = transport_volmask_operators, dims = (1,2,3), condition = masks[j], suffix = suffixes[j]*"tot")
# end

# @info "Merging velocity tuples"

# transport_tuple = NamedTuple{Tuple(transport_names)}(Tuple(transport_outputs))

output_intervals = 5days

output_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/outputs/")

simulation.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                                 dir = output_path,
                                                 schedule = TimeInterval(output_intervals),
                                                 filename = "global_surface_fields",
                                                 indices = (:, :, grid.Nz),
                                                 with_halos = false,
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

simulation.output_writers[:fluxes] = JLD2Writer(ocean.model, fluxes;
                                                dir = output_path,
                                                schedule = TimeInterval(output_intervals),
                                                filename = "fluxes",
                                                overwrite_existing = true)
#=
simulation.output_writers[:ocean_tracer_content] = JLD2Writer(ocean.model, tracer_tuple;
                                                          dir = output_path,
                                                          schedule = TimeInterval(output_intervals),
                                                          filename = "ocean_tracer_content",
                                                          overwrite_existing = true)

simulation.output_writers[:transport] = JLD2Writer(ocean.model, transport_tuple;
                                                          dir = output_path,
                                                          schedule = TimeInterval(output_intervals),
                                                          filename = "mass_transport",
                                                          overwrite_existing = true)
=#
wall_time = Ref(time_ns())

function progress(sim)
    u, v, w = sim.model.ocean.model.velocities
    T = sim.model.ocean.model.tracers.T
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

add_callback!(simulation, progress, IterationInterval(10))

run!(simulation)

simulation.Δt = 20minutes
simulation.stop_time = 11000days

run!(simulation)