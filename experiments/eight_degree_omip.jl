using MPI
MPI.Init()

using ClimaOcean
using ClimaSeaIce
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.OrthogonalSphericalShellGrids
using ClimaOcean.OceanSimulations
using ClimaOcean.ECCO
using ClimaOcean.JRA55
using ClimaOcean.DataWrangling
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium
using Printf
using Dates
using CUDA
using JLD2

function synch!(clock1::Clock, clock2)
    # Synchronize the clocks
    clock1.time = clock2.time
    clock1.iteration = clock2.iteration
    clock1.last_Δt = clock2.last_Δt
end

synch!(model1, model2) = synch!(model1.clock, model2.clock)
restart_iteration = "50000" 

arch = Distributed(GPU(), partition=Partition(y=Oceananigans.DistributedComputations.Equal()), synchronized_communication=true)

Nx = 2880 # longitudinal direction 
Ny = 1440 # meridional direction 
Nz = 100

z_faces = ExponentialDiscretization(Nz, -6000, 0; scale=1800, mutable=true)

const z_surf = z_faces.cᵃᵃᶠ(Nz)

grid = TripolarGrid(arch;
                    size = (Nx, Ny, Nz),
                    z = z_faces,
                    halo = (7, 7, 7))

bottom_height = regrid_bathymetry(grid; minimum_depth=15, major_basins=1, interpolation_passes=55)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

#####
##### A Propgnostic Ocean model
#####

using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, AdvectiveFormulation, IsopycnalSkewSymmetricDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation

buffer_scheme      = WENO(order=5, buffer_scheme=Centered())
momentum_advection = WENOVectorInvariant()
tracer_advection   = WENO(order=7; buffer_scheme)
free_surface       = SplitExplicitFreeSurface(grid; cfl=0.8, fixed_Δt=12minutes)

catke_closure = ClimaOcean.OceanSimulations.default_ocean_closure()  
closure = catke_closure

dataset = EN4Monthly()
date = DateTime(1958, 1, 1)
@inline mask(x, y, z, t) = z ≥ z_surf - 1
Smetadata = Metadata(:salinity; dataset, start_date=date)

FS = DatasetRestoring(Smetadata, Oceananigans.DistributedComputations.child_architecture(arch); rate = 1/30days, mask, time_indices_in_memory=10) 

ocean = ocean_simulation(grid; Δt=1minutes,
                         momentum_advection,
                         tracer_advection,
                         timestepper = :SplitRungeKutta3,
                         free_surface,
                         forcing = (; S = FS),
                         radiative_forcing = nothing,
                         closure)

set!(ocean.model, T=Metadatum(:temperature; dataset, date),
                  S=Metadatum(:salinity;    dataset, date))


#####
##### A Prognostic Sea-ice model
#####

# Default sea-ice dynamics and salinity coupling are included in the defaults
sea_ice = sea_ice_simulation(grid, ocean; advection=WENO(order=7)) 

set!(sea_ice.model, h=Metadatum(:sea_ice_thickness;     dataset=ECCO4Monthly()),
                    ℵ=Metadatum(:sea_ice_concentration; dataset=ECCO4Monthly()))

#####
##### A Prescribed Atmosphere model
#####

dir = "./forcing_data"
dataset = MultiYearJRA55()
backend = JRA55NetCDFBackend(30)

atmosphere = JRA55PrescribedAtmosphere(arch; dir, dataset, backend, include_rivers_and_icebergs=true)
radiation  = Radiation()

#####
##### An ocean-sea ice coupled model
#####

omip = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

include("initialize_omip.jl")

if !isnothing(restart_iteration)
    checkpoint = "eightdegree_iteration$(restart_iteration)_$(MPI.Comm_rank(MPI.COMM_WORLD))_checkpoint.jld2"
    initialize_omip!(omip, checkpoint)

    synch!(atmosphere, ocean.model)
    synch!(omip, ocean.model)
    time_step!(atmosphere, 0)
    atmosphere.clock.iteration -= 1

    @info omip.clock
    @info atmosphere.clock
    @info ocean.model.clock
    @info sea_ice.model.clock
end

omip = Simulation(omip, Δt=1minutes, stop_time=20days) 

# Figure out the outputs....

synch!(omip.model, ocean.model)

wall_time = Ref(time_ns())

function save_datafile(sim; checkpoint=false)
    ocean  = sim.model.ocean
    seaice = sim.model.sea_ice
    suffix = "eightdegree_iteration$(ocean.model.clock.iteration)_$(MPI.Comm_rank(MPI.COMM_WORLD))" 

    MyFloat32 = checkpoint ? Float64 : Float32

    uo = MyFloat32.(Array(interior(ocean.model.velocities.u)))  
    vo = MyFloat32.(Array(interior(ocean.model.velocities.v)))
    wo = MyFloat32.(Array(interior(ocean.model.velocities.w)))
    To = MyFloat32.(Array(interior(ocean.model.tracers.T)))
    So = MyFloat32.(Array(interior(ocean.model.tracers.S)))
    
    eo = if haskey(ocean.model.tracers, :e)
        MyFloat32.(Array(interior(ocean.model.tracers.e)))
    else
        0
    end
    
    ηo = MyFloat32.(Array(interior(ocean.model.free_surface.η)))
    
    ui = MyFloat32.(Array(interior(seaice.model.velocities.u)))
    vi = MyFloat32.(Array(interior(seaice.model.velocities.v)))
    hi = MyFloat32.(Array(interior(seaice.model.ice_thickness)))
    ℵi = MyFloat32.(Array(interior(seaice.model.ice_concentration)))

    if checkpoint
        suffix *= "_checkpoint"
    end

    jldsave(suffix * ".jld2"; uo, vo, wo, To, clock = ocean.model.clock, So, eo, ηo, ui, vi, hi, ℵi)
end

using Statistics

function progress(sim)
    sea_ice = sim.model.sea_ice
    ocean   = sim.model.ocean
    hmax = maximum(sea_ice.model.ice_thickness)
    ℵmax = maximum(sea_ice.model.ice_concentration)
    Tmax = maximum(sim.model.interfaces.atmosphere_sea_ice_interface.temperature)
    Tmin = minimum(sim.model.interfaces.atmosphere_sea_ice_interface.temperature)
    umax = maximum(ocean.model.velocities.u)
    vmax = maximum(ocean.model.velocities.v)
    wmax = maximum(ocean.model.velocities.w)

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("max(h): %.2e m, max(ℵ): %.2e ", hmax, ℵmax)
    msg4 = @sprintf("extrema(T): (%.2f, %.2f) ᵒC, ", Tmax, Tmin)
    msg5 = @sprintf("maximum(u): (%.2f, %.2f, %.2f) m/s, ", umax, vmax, wmax)
    msg6 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg4 * msg5 * msg6

     wall_time[] = time_ns()

     return nothing
end

# And add it as a callback to the simulation.
add_callback!(omip, progress, IterationInterval(10))
add_callback!(omip, sim -> save_datafile(sim), IterationInterval(500))
add_callback!(omip, sim -> save_datafile(sim; checkpoint=true), IterationInterval(5000))

run!(omip)

omip.Δt = 12minutes
omip.stop_time = 58*365days

run!(omip)

