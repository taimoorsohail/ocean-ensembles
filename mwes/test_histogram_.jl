using MPI
using CUDA
using CUDA: @allowscalar
MPI.Init()
atexit(MPI.Finalize)  

using Oceananigans
using NumericalEarth
using CairoMakie
using NumericalEarth.EN4
using NumericalEarth.EN4: download_dataset
using Oceananigans.Units
using Dates
using Oceananigans.DistributedComputations

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

arch = Distributed(CPU(); partition = Partition(y = DistributedComputations.Equal()), synchronized_communication=true)

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

Nx = Integer(40)
Ny = Integer(40)
Nz = Integer(10)

@info "Defining vertical z faces"
depth = -6000.0 # Depth of the ocean in meters
z_faces = ExponentialDiscretization(Nz, depth, 0, mutable=true) # IMPORTANT: WE NEED TO ACCOUNT FOR THIS

const z_surf = z_faces.cᵃᵃᶠ(Nz)
@info "Top grid cell is " * string(abs(round(z_surf))) * "m thick"

@info "Defining tripolar grid"
underlying_grid = TripolarGrid(arch;
                            size = (Nx, Ny, Nz),
                            z = z_faces,
                            halo = (7, 7, 7))

@time bottom_height = regrid_bathymetry(underlying_grid;
                                        minimum_depth = 15,
                                        interpolation_passes = 1, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
                                        major_basins = 4)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

@info "Defining ocean model"

free_surface = SplitExplicitFreeSurface(grid; substeps=10)

@time ocean = ocean_simulation(grid; Δt=1minutes,
                                momentum_advection = nothing,
                                tracer_advection = nothing,
                                free_surface)

set!(ocean.model, T=Metadata(:temperature; dates=first(dates), dataset = dataset, dir=data_path),
                  S=Metadata(:salinity;    dates=first(dates), dataset = dataset, dir=data_path))


Tmin, Tmax = extrema(interior(T_ocean))
Smin, Smax = extrema(interior(S_ocean))

ΔT = 0.1
ΔS = 0.01

T_bins = collect(Tmin:ΔT:(Tmax + ΔT))
S_bins = collect(Smin:ΔS:(Smax + ΔS))

h = Field(Histogram((S=S_ocean, T=T_ocean); bins=(S=S_bins, T=T_bins), weights=:cell_volume))

Vccc = KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.Vᶜᶜᶜ, grid)

@info sum(h), sum(Vccc)
@assert isapprox(sum(h), sum(Vccc); rtol=1e-10, atol=1e-6)

### PLOT ###

fig = Figure(size = (1100, 700))

ax1 = Axis(fig[1, 1], title = "Tripolar TS distribution", xlabel = "Salinity (psu)", ylabel = "Temperature (deg-C)")

hm1 = heatmap!(ax1, log10(h),  label = "TS_distribution", colormap=:blues)

Colorbar(fig[1, 2], hm1, label = "Volume (m3)")

save(figdir * "TS_histogram_test.png", fig, px_per_unit=3)