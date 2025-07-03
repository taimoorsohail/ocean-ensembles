# using MPI
# using CUDA
# using Adapt

# MPI.Init()
# atexit(MPI.Finalize)  

using ClimaOcean
using Oceananigans
using Oceananigans.Units
using CFTime
using Dates
using Printf
using Oceananigans.DistributedComputations
using Oceananigans.Architectures: on_architecture
using JLD2
using OceanEnsembles: basin_mask
using Oceananigans.Operators: Ax, Ay, Az, Δz
using Oceananigans.Fields: ReducedField
using OceanEnsembles

arch = CPU()#Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)
Nx = Integer(360/3)
Ny = Integer(180/3)
Nz = Integer(75)

@info "Defining vertical z faces"

r_faces = exponential_z_faces(; Nz, depth=5000, h=12.43)
# z_faces = Oceananigans.MutableVerticalDiscretization(r_faces)

@info "Defining tripolar grid"

grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = r_faces,
                    halo = (7, 7, 4),
                    first_pole_longitude = 70,
                    north_poles_latitude = 55)

# ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir = data_path)
# ClimaOcean.DataWrangling.download_dataset(ETOPOmetadata)

# @time bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
#                                     minimum_depth = 15,
#                                     major_basins = 1)

# For this bathymetry at this horizontal resolution we need to manually open the Gibraltar strait.
# view(bottom_height, 102:103, 124, 1) .= -400

@info "Defining grid"

# @time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)
@info "Defining closures"

# momentum_advection = WENOVectorInvariant(order=5) 
# tracer_advection = WENO(order=5)

# free_surface = SplitExplicitFreeSurface(grid; substeps=70)

# using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity,
#                                        DiffusiveFormulation

# eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3,
#                                                  skew_flux_formulation=DiffusiveFormulation())
# vertical_mixing = ClimaOcean.OceanSimulations.default_ocean_closure()

# closure = (eddy_closure, vertical_mixing)

@info "Defining ocean simulation"
free_surface       = SplitExplicitFreeSurface(grid; substeps=70)

@time ocean = ocean_simulation(grid;
                        #  momentum_advection,
                        #  tracer_advection,
                         free_surface)
                        #  closure,
                        #  forcing)


radiation  = Radiation(arch)
@time atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(25))

coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=10minutes, stop_time=60days)

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = IterationInterval(1)

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

tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)

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
@info "Tracers"

using Oceananigans.Fields: location, ReducedField
function ocean_tracer_content!(names, ∫outputs; outputs, operator, dims, condition, suffix::AbstractString)
    for key in keys(outputs)
        @show key
        f = outputs[key]
        @time ∫f = Integral(f * operator; dims, condition)

        push!(∫outputs, ∫f)
        push!(names, Symbol(key, suffix))
    end

    onefield = CenterField(first(outputs).grid)
    set!(onefield, 1)
    @info "Integrated volume"
    @time ∫dV = Integral(onefield * operator; dims, condition)
    push!(∫outputs, ∫dV)
    push!(names, Symbol(:dV, suffix))

    return names, ∫outputs
end

j = 1
@time ocean_tracer_content!(tracer_names, tracer_outputs; outputs = tracers, operator = tracer_volmask[1], dims = (1), condition = masks[j][1], suffix = suffixes[j]*"zonal");
    # @time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[2], dims = (1, 2), condition = masks[j][1], suffix = suffixes[j]*"depth");
    # @time ocean_tracer_content!(tracer_names, tracer_outputs; outputs=tracers, operator = tracer_volmask[3], dims = (1, 2, 3), condition = masks[j][1], suffix = suffixes[j]*"tot");

@info "Merging tracer tuples"

tracer_tuple = NamedTuple{Tuple(tracer_names)}(Tuple(tracer_outputs))

output_intervals = TimeInterval(20minutes)
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")

@time simulation.output_writers[:ocean_tracer_content] = JLD2Writer(ocean.model, tracer_tuple;
                                                          dir = output_path,
                                                          schedule = output_intervals,
                                                          filename = "ocean_tracer_content_test_iteration" * string(iteration),
                                                          overwrite_existing = true)

# simulation.Δt = 5minutes
# simulation.stop_time = target_time # 1 year

run!(simulation)