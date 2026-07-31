using Oceananigans
using Oceananigans.Fields: ConstantField, ZeroField
using Oceananigans.Units
using NumericalEarth
using ClimaSeaIce
using ClimaSeaIce: SeaIceModel, ConductiveFlux, sea_ice_slab_thermodynamics
using ClimaSeaIce.SeaIceThermodynamics.HeatBoundaryConditions: IceWaterThermalEquilibrium
using ClimaSeaIce.SeaIceThermodynamics: PrescribedTemperature

using CairoMakie
using Printf
using Statistics
using CUDA

arch = GPU()

@info "Running sea ice melt checkerboard MWE on $arch"
# Small, fast coastal-channel style setup:
# periodic in x, bounded in y, shallow in z, with a compact ice patch hugging the south wall.
Nx = 150
Ny = 150
Nz = 15
@info "Grid size: $(Nx) x $(Ny) x $(Nz)"
Lx = 6.4kilometers
Ly = 3.2kilometers
Lz = 16meters
@info "Domain size: $(Lx) x $(Ly) x $(Lz)"
Δt = 10seconds
stop_time = 40minutes
save_interval = 10
@info   "Time step: $(Δt), stop time: $(stop_time), save interval: $(save_interval)"
S₀ = 30.0
T₀ = 1.0
@info "Initial conditions: T = $(T₀)°C, S = $(S₀) psu"
ice_grid = RectilinearGrid(arch;
                           size = (Nx, Ny),
                           x = (-Lx / 2, Lx / 2),
                           y = (0, Ly),
                           halo = (6, 6),
                           topology = (Periodic, Bounded, Flat))
@info   "Ice grid: size = $(size(ice_grid)), x = $(xnodes(ice_grid, Center())[1]) to $(xnodes(ice_grid, Center())[end]), y = $(ynodes(ice_grid, Center())[1]) to $(ynodes(ice_grid, Center())[end])"
ocean_grid = RectilinearGrid(arch;
                             size = (Nx, Ny, Nz),
                             x = (-Lx / 2, Lx / 2),
                             y = (0, Ly),
                             z = ExponentialDiscretization(Nz, -Lz, 0, mutable=true),
                             halo = (6, 6, 3),
                             topology = (Periodic, Bounded, Bounded))

ice_ocean_heat_flux = Field{Center, Center, Nothing}(ice_grid)
Qᵀ = Field{Center, Center, Nothing}(ice_grid)
Qˢ = Field{Center, Center, Nothing}(ice_grid)
@info "Initialized ice-ocean interface heat flux fields on ice grid with size $(size(ice_ocean_heat_flux))"
boundary_conditions = (
    T = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵀ)),
    S = FieldBoundaryConditions(top = FluxBoundaryCondition(Qˢ)),
)
@info "Defined boundary conditions for ocean temperature and salinity with fluxes Qᵀ and Qˢ at the top"
ocean = ocean_simulation(ocean_grid;
                                timestepper=:SplitRungeKutta3,
                                closure = (ScalarDiffusivity(ν = 1e-4, κ = 1e-5)))

u, v, w = ocean.model.velocities
surface_velocities = (u = view(u, :, :, Nz),
                      v = view(v, :, :, Nz),
                      w = ZeroField())
@info "Defined surface velocity fields on the top layer of the ocean grid"
ice = sea_ice_simulation(ice_grid, ocean)
@info "Created sea ice simulation with grid size $(size(ice_grid))"
ice_patch(x, y) = abs(x) < 0.8kilometers && abs(y - Ly / 2) < 0.8kilometers
Tᵢ(x, y, z) = T₀

Sᵢ(x, y, z) = S₀

uᵢ(x, y, z) = 1e-2 * sin(2π * x / Lx) * cos(2π * y / Ly)
vᵢ(x, y, z) = 1e-2 * cos(2π * x / Lx) * sin(2π * y / Ly)

hᵢ(x, y) = ice_patch(x, y) ? 0.25 : 0.0
ℵᵢ(x, y) = ice_patch(x, y) ? 1.0 : 0.0
@info "Defined initial conditions for ocean temperature, salinity, and velocities, and for ice thickness and concentration"
set!(ocean.model, u = uᵢ, v = vᵢ, T = Tᵢ, S = Sᵢ)
set!(ice.model, h = hᵢ, ℵ = ℵᵢ)
@info "Set initial conditions in ocean and ice models"
coupled_model = OceanSeaIceModel(ice, ocean)
simulation = Simulation(coupled_model; Δt, stop_time, verbose = false)
h = ice.model.ice_thickness
T = ocean.model.tracers.T
Q_interface = ice.model.external_heat_fluxes.bottom
@info "Defined convenient references to ice thickness, ocean temperature, and ice-ocean interface heat flux fields"
ht = Matrix{Float64}[]
Tt = Matrix{Float64}[]
Qt = Matrix{Float64}[]
wt = Matrix{Float64}[]
times = Float64[]
max_heat_flux = Float64[]
max_melt_rate = Float64[]
checkerboard_ratio = Float64[]
advective_cfls = Float64[]

previous_h = Ref{Union{Nothing, Matrix{Float64}}}(nothing)
previous_time = Ref{Float64}(0.0)
@info "Initialized arrays and references for storing time series of ice thickness, ocean temperature, heat flux, vertical velocity, times, max heat flux, max melt rate, and checkerboard ratio during the simulation"
neighbor_mean_abs(A) = mean(abs, A[1:end-1, :] .+ A[2:end, :])
@info "Defined a helper function to compute the mean absolute value of neighboring elements in a 2D array, used for checkerboard ratio calculation"
function save_state(sim)
    hn = Array(interior(h, :, :, 1))
    Tn = Array(interior(T, :, :, Nz))
    Qn = Array(interior(Q_interface, :, :, 1))
    wn = Array(interior(w, :, :, Nz))
    t = time(sim)

    push!(ht, hn)
    push!(Tt, Tn)
    push!(Qt, Qn)
    push!(wt, wn)
    push!(times, t)
    push!(max_heat_flux, maximum(abs, Qn))
    push!(advective_cfls, AdvectiveCFL(sim.Δt)(sim.model.ocean.model))

    if isnothing(previous_h[])
        push!(max_melt_rate, 0.0)
    else
        Δh = (hn .- previous_h[]) ./ (t - previous_time[])
        push!(max_melt_rate, maximum(-Δh))
    end

    mean_abs_w = mean(abs, wn)
    ratio = mean_abs_w == 0 ? 1.0 : neighbor_mean_abs(wn) / (2 * mean_abs_w)
    push!(checkerboard_ratio, ratio)

    previous_h[] = hn
    previous_time[] = t

    return nothing
end

function progress(sim)
    n = length(times)
    n == 0 && return nothing

    msg = @sprintf("iter: %4d, time: %8s, max|Q_interface|: %8.2f W m⁻², max melt: %.3e m s⁻¹, checkerboard ratio: %.3f, advective CFL: %.2f",
                   iteration(sim),
                   prettytime(sim),
                   max_heat_flux[n],
                   max_melt_rate[n],
                   checkerboard_ratio[n],
                   advective_cfls[n])
    @info msg
    return nothing
end

simulation.callbacks[:save_state] = Callback(save_state, IterationInterval(save_interval))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(save_interval))

save_state(simulation)
@info "Starting simulation..."
run!(simulation)

@info @sprintf("Peak max|Q_interface| = %.2f W m⁻²", maximum(max_heat_flux))
@info @sprintf("Peak max melt rate   = %.3e m s⁻¹", maximum(max_melt_rate))
@info @sprintf("Minimum checkerboard ratio = %.3f", minimum(checkerboard_ratio))
@info @sprintf("Peak advective CFL = %.2f", maximum(advective_cfls))

Nt = length(times)
x = xnodes(ice_grid, Center()) ./ 1e3
y = ynodes(ice_grid, Center()) ./ 1e3

hlim = (minimum(minimum, ht), maximum(maximum, ht))
Tlim = (minimum(minimum, Tt), maximum(maximum, Tt))
Qlim_val = maximum(maximum(abs, Qfield) for Qfield in Qt)
Qlim = (-Qlim_val, Qlim_val)
wlim_val = maximum(maximum(abs, wfield) for wfield in wt)
wlim = (-wlim_val, wlim_val)

fig = Figure(size = (1400, 900))
frame = Observable(1)

current_time = @lift prettytime(times[$frame])
current_summary = @lift @sprintf("max |Q_interface| = %.1f W m⁻²   max melt = %.3e m s⁻¹   checkerboard ratio = %.3f   advective CFL = %.2f",
                                 max_heat_flux[$frame],
                                 max_melt_rate[$frame],
                                 checkerboard_ratio[$frame],
                                 advective_cfls[$frame])

Label(fig[0, 1:2], @lift("Warm-under-ice MWE at " * $current_time), fontsize = 26)

axh = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "y (km)", title = "Ice thickness (m)")
axT = Axis(fig[1, 2], xlabel = "x (km)", ylabel = "y (km)", title = "Surface ocean temperature (°C)")
axQ = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "y (km)", title = "Interface heat flux (W m⁻²)")
axw = Axis(fig[2, 2], xlabel = "x (km)", ylabel = "y (km)", title = "Top-cell vertical velocity (m s⁻¹)")

hm_h = heatmap!(axh, x, y, @lift(ht[$frame]), colormap = :ice, colorrange = hlim)
hm_T = heatmap!(axT, x, y, @lift(Tt[$frame]), colormap = :thermal, colorrange = Tlim)
hm_Q = heatmap!(axQ, x, y, @lift(Qt[$frame]), colormap = :balance, colorrange = Qlim)
hm_w = heatmap!(axw, x, y, @lift(wt[$frame]), colormap = :balance, colorrange = wlim)

Colorbar(fig[1, 3], hm_h)
Colorbar(fig[1, 4], hm_T)
Colorbar(fig[2, 3], hm_Q)
Colorbar(fig[2, 4], hm_w)

Label(fig[3, 1:2], current_summary, fontsize = 20, tellwidth = false)

animation_path = joinpath(@__DIR__, "sea_ice_melt_checkerboard_mwe.mp4")
CairoMakie.record(fig, animation_path, 1:Nt, framerate = 10) do n
    frame[] = n
end

@info "Saved animation to $animation_path"
