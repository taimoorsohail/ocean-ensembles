using MPI
using CUDA

MPI.Init()
atexit(MPI.Finalize)  

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Printf
using Dates
using Oceananigans.Architectures: on_architecture

using ClimaSeaIce
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium

using ClimaOcean
using ClimaOcean.ECCO
using JLD2

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")

arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=false)

function analytical_immersed_tripolar_grid(underlying_grid::TripolarGrid; radius = 5, active_cells_map = false) # degrees
    λp = underlying_grid.conformal_mapping.first_pole_longitude
    φp = underlying_grid.conformal_mapping.north_poles_latitude
    φm = underlying_grid.conformal_mapping.southernmost_latitude

    Lz = underlying_grid.Lz

    # We need a bottom height field that ``masks'' the singularities
    bottom_height(λ, φ) = ((abs(λ - λp) < radius)       & (abs(φp - φ) < radius)) |
                          ((abs(λ - λp - 180) < radius) & (abs(φp - φ) < radius)) | (φ < φm) ? 0 : - Lz

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map)

    return grid
end

Nx, Ny, Nz = 100, 100, 50
Lx, Ly = 100, 100

@info "Defining vertical z faces"

depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialDiscretization(Nz, depth, 0) 

@info "Creating grid"

underlying_grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (7, 7, 4))

@info "Defining grid"

grid = analytical_immersed_tripolar_grid(underlying_grid; active_cells_map=true)

@info "Creating free surface"
free_surface = SplitExplicitFreeSurface(grid; cfl=0.7, fixed_Δt=10minutes)

@info "Creating model"

@time ocean = ocean_simulation(grid; Δt=1minutes, free_surface, timestepper = :SplitRungeKutta3)

@info "Creating simulation"

sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7))

set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dataset=ECCO4Monthly(), dir=data_path),
                    ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly(), dir=data_path))

@info "Defining Atmospheric state"

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(100), include_rivers_and_icebergs=true)

@info "Defining coupled model"
@time coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

simulation = Simulation(coupled_model; Δt=10, verbose=false, stop_time=2hours)

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = TimeInterval(1days)

import Oceananigans.Diagnostics: CFL
(c::CFL)(sim::Simulation) = c(sim.model)
(c::CFL)(sim::Simulation{<:ClimaOcean.OceanSeaIceModels.OceanSeaIceModel}) = c(sim.model.ocean.model)

function progress_message(sim)
    η = sim.model.ocean.model.free_surface.η
    u, v, w = sim.model.ocean.model.velocities
    T, S = sim.model.ocean.model.tracers
    cfl = AdvectiveCFL(sim.Δt)

    Trange = (maximum((T)), minimum((T)))
    Srange = (maximum((S)), minimum((S)))
    ηrange = (maximum((η)), minimum((η)))

    umax = (maximum(abs, (u)),
            maximum(abs, (v)),
            maximum(abs, (w)))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s,", prettytime(sim), Oceananigans.iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max|u|: (%.2e, %.2e, %.2e) m s⁻¹, ", umax...)
    msg3 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Trange...)
    msg4 = @sprintf("extrema(S): (%.2f, %.2f) g/kg, ", Srange...)
    msg6 = @sprintf("extrema(η): (%.2f, %.2f) m, ", ηrange...)
    msg7 = @sprintf("wall time: %s \n", prettytime(step_time))
    msg8 = @sprintf("SYPD: %.2f \n", (24*3600)/step_time/365)
    msg5 = @sprintf("CFL: %.2f \n", getfield(cfl, 1))

    @info msg1 * msg2 * msg3 * msg4 * msg6 * msg7 * msg8 * msg5

    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress_message, IterationInterval(40))

@info "Running simulation"
run!(simulation)