using Oceananigans
using NumericalEarth
using CairoMakie
using NumericalEarth.ECCO
using NumericalEarth.ECCO: download_dataset

using Oceananigans.Units
using Dates
using Oceananigans.DistributedComputations
using Oceananigans.Operators: Ax, Ay, Az, Δz

data_path = expanduser("/g/data/v46/txs156/ocean-ensembles/data/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

arch = CPU()

# ### ECCO files
@info "Downloading/checking input data"

dataset     = ECCO4Monthly() # Other options include ECCO2Monthly(), ECCO4Monthly() or ECCO2Daily()

temperature = Metadata(:temperature; dates = DateTime(1998, 1, 1), dataset = dataset, dir=data_path)
salinity    = Metadata(:salinity;    dates = DateTime(1998, 1, 1), dataset = dataset, dir=data_path)
u           = Metadata(:u_velocity;  dates = DateTime(1998, 1, 1), dataset = dataset, dir=data_path)
v           = Metadata(:v_velocity;  dates = DateTime(1998, 1, 1), dataset = dataset, dir=data_path)

download_dataset(temperature)
download_dataset(salinity)
download_dataset(u)
download_dataset(v)

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

set!(ocean.model, T = Metadata(:temperature;  dates=DateTime(1998, 1, 1), dataset = dataset, dir = data_path),
                  S = Metadata(:salinity   ;  dates=DateTime(1998, 1, 1), dataset = dataset, dir = data_path),
                  u = Metadata(:u_velocity ;  dates=DateTime(1998, 1, 1), dataset = dataset, dir = data_path),
                  v = Metadata(:v_velocity ;  dates=DateTime(1998, 1, 1), dataset = dataset, dir = data_path))

T_ocean = ocean.model.tracers.T
S_ocean = ocean.model.tracers.S

Tmin, Tmax = extrema(interior(T_ocean))
Smin, Smax = extrema(interior(S_ocean))

ΔT = 0.1
ΔS = 0.01

T_bins = collect(Tmin:ΔT:(Tmax + ΔT))
S_bins = collect(Smin:ΔS:(Smax + ΔS))

h = Field(Histogram((S=S_ocean, T=T_ocean); bins=(S=S_bins, T=T_bins), weights=:count, method = :integral))

Vccc = KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.Vᶜᶜᶜ, grid)

@info sum(h), sum(Vccc)
@assert isapprox(sum(h), sum(Vccc); rtol=1e-10, atol=1e-6)

### PLOT ###

fig = Figure(size = (1100, 700))

ax1 = Axis(fig[1, 1], title = "Tripolar TS distribution", xlabel = "Salinity (psu)", ylabel = "Temperature (deg-C)")

hm1 = heatmap!(ax1, log10(h),  label = "TS_distribution", colormap=:blues)

Colorbar(fig[1, 2], hm1, label = "Volume (m3)")

save(figdir * "TS_histogram_test.png", fig, px_per_unit=3)

SdA = Field(Histogram((; T=T_ocean); bins=(T=T_bins), weights=S_ocean, dims=(1,2), method=:integral))
A_tot = Field(Histogram((; T=T_ocean); bins=(T=T_bins), weights=:count,   dims=(1,2), method=:integral))

S_mean = Field(SdA / A_tot)
S_avg  = Field(Histogram((; T=T_ocean); bins=(T=T_bins), weights=S_ocean, dims=(1,2), method=:average))

compute!(S_mean); compute!(S_avg); compute!(A_tot)

A = Array(interior(S_mean))
B = Array(interior(S_avg))
M = Array(interior(A_tot)) .> 0

@assert all(isapprox.(A[M], B[M]; rtol=1e-12, atol=1e-12))

