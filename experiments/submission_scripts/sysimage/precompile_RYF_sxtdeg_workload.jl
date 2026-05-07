# Cheap representative workload for building a reusable RYF_sxtdeg sysimage.
# Run by PackageCompiler during sysimage creation. Keep this small, but keep the
# object types close to the production script so Julia compiles the hot methods.

using MPI
using CUDA
using NumericalEarth
using NumericalEarth.EN4
using NumericalEarth.ECCO
using NumericalEarth.DataWrangling.ETOPO
using ClimaSeaIce
using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Oceananigans.Operators: Ax, Ay, Az, Δz
using Oceananigans.Fields: ReducedField
using Oceananigans.Architectures: on_architecture
using OceanEnsembles
using CFTime
using Dates
using Printf
using Glob
using JLD2

if !MPI.Initialized()
    MPI.Init()
end

arch_name = get(ENV, "RYF_SYSIMAGE_ARCH", "GPU")
arch = if arch_name == "GPU"
    Distributed(GPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication = true)
elseif arch_name == "CPU"
    Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication = true)
else
    error("RYF_SYSIMAGE_ARCH must be GPU or CPU, got $(arch_name)")
end

# These dimensions are tiny compared with production. Method compilation mostly
# depends on types, not global grid size, so this keeps the workload cheap.
total_ranks = MPI.Comm_size(MPI.COMM_WORLD)
Nx = parse(Int, get(ENV, "RYF_SYSIMAGE_NX", "24"))
Ny_per_rank = parse(Int, get(ENV, "RYF_SYSIMAGE_NY_PER_RANK", "32"))
Ny = max(total_ranks, total_ranks * Ny_per_rank)
Nz = parse(Int, get(ENV, "RYF_SYSIMAGE_NZ", "8"))

data_path = expanduser(get(ENV, "RYF_DATA_PATH", "/home/tsohail/uom/ocean-ensembles/data/"))
output_path = mktempdir()

dates = [DateTime(1991, 1, 1), DateTime(1991, 2, 1)]
dataset = EN4Monthly()
temperature = Metadata(:temperature; dates, dataset, dir = data_path)
salinity = Metadata(:salinity; dates, dataset, dir = data_path)

z_faces = ExponentialDiscretization(Nz, -500.0, 0, mutable = true)
const z_surf = z_faces.cᵃᵃᶠ(Nz)

underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), z = z_faces, halo = (7, 7, 7))

# Avoid downloading/regridding ETOPO during sysimage creation. This still builds
# the immersed-boundary grid type used by the production script.
bottom_height = Field{Center, Center, Nothing}(underlying_grid)
set!(bottom_height, -500)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map = true)

const restoring_rate = 1 / 30days
@inline mask(x, y, z, t) = z >= z_surf - 1
forcing = (; S = DatasetRestoring(salinity, grid; mask, rate = restoring_rate, time_indices_in_memory = 10))

catke_closure = NumericalEarth.Oceans.default_ocean_closure()
closure = (catke_closure, VerticalScalarDiffusivity(κ = 1e-5, ν = 1e-4))
free_surface = SplitExplicitFreeSurface(grid; substeps = 70)
momentum_advection = WENOVectorInvariant()
tracer_advection = WENO(order = 7)

ocean = ocean_simulation(grid; Δt = 1minutes,
                         momentum_advection,
                         tracer_advection,
                         timestepper = :SplitRungeKutta3,
                         free_surface,
                         forcing,
                         radiative_forcing = nothing,
                         closure)

# Touch initialization/restoring machinery without running a long production case.
set!(ocean.model, T = Metadata(:temperature; dates = first(dates), dataset, dir = data_path),
                  S = Metadata(:salinity; dates = first(dates), dataset, dir = data_path))

sea_ice = sea_ice_simulation(grid, ocean; advection = WENO(order = 7))
set!(sea_ice.model, h = Metadatum(:sea_ice_thickness; dataset = ECCO4Monthly(), dir = data_path),
                    ℵ = Metadatum(:sea_ice_concentration; dataset = ECCO4Monthly(), dir = data_path))

radiation = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory = 100, include_rivers_and_icebergs = true)
coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt = 10minutes)

tracers = ocean.model.tracers
velocities = ocean.model.velocities
outputs = merge(tracers, velocities)
surface_height = (; surface_height = ocean.model.free_surface.displacement)
surface_forcing = (; heat_flux = Field(net_ocean_heat_flux(simulation.model)),
                    fw_flux = Field(net_ocean_freshwater_flux(simulation.model)))

tot_integral = Symbol[]
tot_integral_outputs = Field[]
surf_integral = Symbol[]
surf_integral_outputs = Field[]
vert_integral = Symbol[]
vert_integral_outputs = Field[]

for key in keys(outputs)
    f = outputs[key]
    f_tot = Field(Integral(f, dims = (1, 2, 3)))
    f_vert = Field(Integral(f, dims = (1, 2)))

    push!(tot_integral_outputs, f_tot)
    push!(tot_integral, Symbol(key, "_totintegral"))

    push!(vert_integral_outputs, f_vert)
    push!(vert_integral, Symbol(key, "_vertintegral"))
end

V_ccc = KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.Vᶜᶜᶜ, grid)
V_fcc = KernelFunctionOperation{Face, Center, Center}(Oceananigans.Operators.Vᶠᶜᶜ, grid)
V_cfc = KernelFunctionOperation{Center, Face, Center}(Oceananigans.Operators.Vᶜᶠᶜ, grid)

totint_vol_c = sum(V_ccc, dims = (1, 2, 3))
totint_vol_x = sum(V_fcc, dims = (1, 2, 3))
totint_vol_y = sum(V_cfc, dims = (1, 2, 3))

vertint_vol_c = sum(V_ccc, dims = (1, 2))
vertint_vol_x = sum(V_fcc, dims = (1, 2))
vertint_vol_y = sum(V_cfc, dims = (1, 2))

cumulative_tuple = NamedTuple{Tuple(tot_integral)}(Tuple(tot_integral_outputs))
cumulative_vert_tuple = NamedTuple{Tuple(vert_integral)}(Tuple(vert_integral_outputs))
cumulative_tuple_vol = (; total_volume_c = totint_vol_c,
                          total_volume_x = totint_vol_x,
                          total_volume_y = totint_vol_y)
cumulative_vert_tuple_vol = (; vert_volume_c = vertint_vol_c,
                               vert_volume_x = vertint_vol_x,
                               vert_volume_y = vertint_vol_y)

global_outputs = merge(cumulative_tuple, cumulative_vert_tuple,
                       cumulative_tuple_vol, cumulative_vert_tuple_vol)

# Compile output writer and checkpoint construction paths against temporary files.
run_id = "0001"
depths = [0, -100, -500, -1000, -2000]

for (ind, depth) in enumerate(depths)
    _, ind_pln = findmin(abs.(grid.z.cᵃᵃᶜ[1:Nz] .- depth))
    ocean.output_writers[Symbol("sysimage_plane", ind)] = JLD2Writer(ocean.model, outputs;
        dir = output_path,
        schedule = IterationInterval(1),
        filename = "global_$(ind_pln)_fields_sxtdeg_RYF_run$(run_id)",
        indices = (:, :, ind_pln),
        with_halos = false,
        overwrite_existing = true,
        array_type = Array{Float32})
end

ocean.output_writers[:SSH] = JLD2Writer(ocean.model, surface_height;
    dir = output_path,
    schedule = IterationInterval(1),
    filename = "global_ssh_fields_sxtdeg_RYF_run$(run_id)",
    with_halos = false,
    overwrite_existing = true,
    array_type = Array{Float32})

simulation.output_writers[:surface_fluxes] = JLD2Writer(simulation.model, surface_forcing;
    dir = output_path,
    schedule = IterationInterval(1),
    filename = "global_surface_fluxes_sxtdeg_RYF_run$(run_id)",
    with_halos = false,
    overwrite_existing = true,
    array_type = Array{Float32})

ocean.output_writers[:integral] = JLD2Writer(ocean.model, global_outputs;
    dir = output_path,
    schedule = AveragedTimeInterval((365 / 48)days),
    filename = "global_tot_integrals_sxtdeg_RYF_run$(run_id)",
    overwrite_existing = true)

simulation.output_writers[:sysimage_checkpoint] = Checkpointer(coupled_model;
    schedule = TimeInterval((365 / 12)days),
    dir = output_path,
    prefix = "sysimage_checkpoint_rank$(Integer(arch.local_rank))",
    overwrite_existing = true,
    cleanup = false)

# One zero-duration step is enough to force final model/simulation setup paths.
simulation.stop_iteration = 0
run!(simulation)

MPI.Barrier(MPI.COMM_WORLD)
@info "RYF_sxtdeg sysimage precompile workload completed" arch_name Nx Ny Nz total_ranks
