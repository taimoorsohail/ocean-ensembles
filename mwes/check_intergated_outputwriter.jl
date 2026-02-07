using CUDA
using Oceananigans

using ClimaOcean

using ClimaOcean.EN4
using ClimaOcean.ECCO
using ClimaOcean.EN4: download_dataset
using ClimaOcean.DataWrangling.ETOPO

using ClimaSeaIce
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium

using Oceananigans.Units

using CFTime
using Dates
using Printf
using Glob 
using JLD2

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

# Argument is provided by the submission script!

arch = GPU()

# ### ECCO files
@info "Downloading/checking input data"

dates = vcat(collect(DateTime(1991, 1, 1): Month(1): DateTime(1991, 5, 1)),
             collect(DateTime(1990, 5, 1): Month(1): DateTime(1990, 12, 1)))

@info "We download the 1990-1991 data for an RYF implementation"

dataset = EN4Monthly() # Other options include ECCO2Monthly(), ECCO4Monthly() or ECCO2Daily()

temperature = Metadata(:temperature; dates, dataset = dataset, dir=data_path)
salinity    = Metadata(:salinity;    dates, dataset = dataset, dir=data_path)

download_dataset(temperature)
download_dataset(salinity)


# ### Grid and Bathymetry
@info "Defining grid"

Nx = Integer(360)
Ny = Integer(180)
Nz = Integer(75)

@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialDiscretization(Nz, depth, 0, mutable=true) # IMPORTANT: WE NEED TO ACCOUNT FOR THIS

@info "Defining tripolar grid"
underlying_grid = TripolarGrid(arch;
                            size = (Nx, Ny, Nz),
                            z = z_faces,
                            halo = (7, 7, 7))

@info "Defining bottom bathymetry"

ETOPOmetadata = Metadatum(:bottom_height, dataset=ETOPO2022(), dir = data_path)
ClimaOcean.DataWrangling.download_dataset(ETOPOmetadata)

@time bottom_height = regrid_bathymetry(underlying_grid, ETOPOmetadata;
                                minimum_depth = 15,
                                interpolation_passes = 1, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
                                major_basins = 4)

@time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)


@info "Defining closures"

catke_closure = ClimaOcean.Oceans.default_ocean_closure()  #RiBasedVerticalDiffusivity()#

closure = (catke_closure, VerticalScalarDiffusivity(κ=1e-5, ν=1e-4))

# ### Ocean simulation
# Now we bring everything together to construct the ocean simulation.
# We use a split-explicit timestepping with 30 substeps for the barotropic
# mode.

@info "Defining free surface"
# output number of substeps
# free_surface = SplitExplicitFreeSurface(grid; cfl=0.7, fixed_Δt=12minutes)

free_surface = SplitExplicitFreeSurface(grid; substeps=70)

momentum_advection = WENOVectorInvariant()
tracer_advection   = WENO(order = 7)

@info "Defining ocean model"

@time ocean = ocean_simulation(grid; Δt=1minutes,
                         momentum_advection,
                         tracer_advection,
                         timestepper = :SplitRungeKutta3,
                         free_surface,
                         radiative_forcing = nothing,
                         closure)

                         # ### Initial condition

# We initialize the ocean from the ECCO state estimate.

@info "Initialising with EN4"

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

simulation = Simulation(coupled_model; Δt=10minutes)

@info "Defining messenger"

wall_time = Ref(time_ns())

callback_interval = IterationInterval(1)

function progress(sim)
    η = sim.model.ocean.model.free_surface.displacement
    u, v, w = sim.model.ocean.model.velocities
    T, S = sim.model.ocean.model.tracers

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
    msg8 = @sprintf("SYPD: %.2f \n", (10*sim.Δt)/step_time/365)

    @info msg1 * msg2 * msg3 * msg4 * msg6 * msg7 * msg8

    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, callback_interval)

################################### START OUTPUTTING ######################################

@info "Defining output variables"

tracers = ocean.model.tracers
velocities = ocean.model.velocities

outputs = merge(tracers, velocities)

surface_height = (; surface_height = ocean.model.free_surface.displacement)
surface_forcing = (; T_surf = ocean.model.tracers.T.boundary_conditions.top.condition, 
                    S_surf = ocean.model.tracers.S.boundary_conditions.top.condition)

outputs_surf = merge(surface_height, surface_forcing)

@info "Defining total integral outputs"

tot_integral = Symbol[]
tot_integral_outputs = Field[]

surf_integral = Symbol[]
surf_integral_outputs = Field[]

vert_integral = Symbol[]
vert_integral_outputs = Field[]

for key in keys(outputs)
    @show key
    f = outputs[key]
    f_tot = Field(Integral(f, dims = (1,2,3)))
    f_vert = Field(Integral(f, dims = (1,2)))

    push!(tot_integral_outputs, f_tot)
    push!(tot_integral, Symbol(key, "_totintegral"))

    push!(vert_integral_outputs, f_vert)
    push!(vert_integral, Symbol(key, "_vertintegral"))
end

@info "Defining surface integral outputs"

for key in keys(surface_forcing)
    f_surf = surface_forcing[key]
    surf_tot = Field(Integral(f_surf, dims = (1,2,3)))
    push!(surf_integral_outputs, surf_tot)
    push!(surf_integral, Symbol(key, "_surfintegral"))
end

V_ccc = KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.Vᶜᶜᶜ, grid)
V_fcc = KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.Vᶠᶜᶜ, grid)
V_cfc = KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.Vᶜᶠᶜ, grid)

@info "Defining total volume integrals"
totint_vol_c = Field(Integral(V_ccc, dims = (1,2,3)))
totint_vol_x = Field(Integral(V_fcc, dims = (1,2,3)))
totint_vol_y = Field(Integral(V_cfc, dims = (1,2,3)))

tot_integral_volumes = [totint_vol_c, totint_vol_x, totint_vol_y]
tot_integral_volume_symbols = [:total_volume_c, :total_volume_x, :total_volume_y]

@info "Defining vertical volume integrals"
vertint_vol_c = Field(Integral(V_ccc, dims = (1,2)))
vertint_vol_x = Field(Integral(V_fcc, dims = (1,2)))
vertint_vol_y = Field(Integral(V_cfc, dims = (1,2)))

vert_integral_volumes = [vertint_vol_c, vertint_vol_x, vertint_vol_y]
vert_integral_volume_symbols = [:vert_volume_c, :vert_volume_x, :vert_volume_y]

@info "Defining integral tuples"

cumulative_tuple = NamedTuple{Tuple(tot_integral)}(Tuple(tot_integral_outputs))
cumulative_vert_tuple = NamedTuple{Tuple(vert_integral)}(Tuple(vert_integral_outputs))

cumulative_tuple_vol = NamedTuple{Tuple(tot_integral_volume_symbols)}(Tuple(tot_integral_volumes))
cumulative_vert_tuple_vol = NamedTuple{Tuple(vert_integral_volume_symbols)}(Tuple(vert_integral_volumes))

global_outputs = merge(cumulative_tuple, cumulative_vert_tuple,
                       cumulative_tuple_vol, cumulative_vert_tuple_vol)

@info "Defining all integrals"

@time ocean.output_writers[:integral] = JLD2Writer(ocean.model, global_outputs;
                                            dir = output_path,
                                            schedule = IterationInterval(1),
                                            filename = "test_integration",
                                            overwrite_existing = true)

################################### END OUTPUTTING ######################################

@info "Running Simulation"
simulation.stop_iteration = 10 

run!(simulation)