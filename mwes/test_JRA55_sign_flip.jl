using CairoMakie
using NumericalEarth
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: FractionalIndices, parent
using Oceananigans.Fields: interpolate as oc_interpolate
using Oceananigans.Grids: halo_size, φnode
using Oceananigans.Operators: intrinsic_vector, rotation_angle
using Oceananigans.OutputReaders: cpu_interpolating_time_indices
using Oceananigans.Operators: Δyᶠᶜᶜ, Δxᶜᶠᶜ

const Nx = 120
const Ny = 60
const Nz = 1
const figdir = expanduser("/home/tsohail/uom/ocean-ensembles/figures/")

grid = TripolarGrid(CPU();
                    size = (Nx, Ny, Nz),
                    z = (-1, 0),
                    halo = (7, 7, 7))

ocean = ocean_simulation(grid;
                         free_surface = SplitExplicitFreeSurface(grid, substeps = 20))

atmosphere = JRA55PrescribedAtmosphere(CPU(); time_indices_in_memory = 2)
radiation  = JRA55PrescribedRadiation(CPU(); time_indices_in_memory = 2)

coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)

exchanger = coupled_model.interfaces.exchanger.atmosphere
regridder = exchanger.regridder
u_field_atmos = exchanger.state.u
v_field_atmos = exchanger.state.v
fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes
net_ocean_fluxes = coupled_model.interfaces.net_fluxes.ocean

ua = atmosphere.velocities.u
va = atmosphere.velocities.v
times = ua.times
time_indexing = ua.time_indexing
backend = ua.backend
t = coupled_model.clock.time
time_itp = cpu_interpolating_time_indices(CPU(), times, time_indexing, t)

Hx, Hy, Hz = halo_size(grid)
i_indices = Hx + 1:Hx + Nx
j_south = Hy + Ny - 1
j_seam  = Hy + Ny
j_halo  = Hy + Ny + 1
k = 1
kN = size(grid, 3)

logical_row(field, j_row) = Array(@view parent(field)[i_indices, j_row, 1])

function halo_filled_row(field, j_row)
    scratch = deepcopy(field)
    fill_halo_regions!(scratch)
    return logical_row(scratch, j_row)
end

function raw_rotation_terms(ii, jj)
    φpp = φnode(ii + 1, jj + 1, 1, grid, Face(), Face(), Center())
    φpm = φnode(ii + 1, jj,     1, grid, Face(), Face(), Center())
    φmp = φnode(ii,     jj + 1, 1, grid, Face(), Face(), Center())
    φmm = φnode(ii,     jj,     1, grid, Face(), Face(), Center())

    Δyp = Δyᶠᶜᶜ(ii + 1, jj,     1, grid)
    Δym = Δyᶠᶜᶜ(ii,     jj,     1, grid)
    Δxp = Δxᶜᶠᶜ(ii,     jj + 1, 1, grid)
    Δxm = Δxᶜᶠᶜ(ii,     jj,     1, grid)

    Rcos1 = ifelse(Δyp == 0, zero(grid), deg2rad(φpp - φpm) / Δyp)
    Rcos2 = ifelse(Δym == 0, zero(grid), deg2rad(φmp - φmm) / Δym)
    Rcosθ = (Rcos1 + Rcos2) / 2
    Rsinθ = -(deg2rad(φpp - φmp) / Δxp + deg2rad(φpm - φmm) / Δxm) / 2
    R = sqrt(Rcosθ^2 + Rsinθ^2)
    θ = rotation_angle(ii, jj, grid)

    return (; φpp, φpm, φmp, φmm, Δyp, Δym, Δxp, Δxm, Rcos1, Rcos2, Rcosθ, Rsinθ, R, θ)
end

function row_diagnostics(j_row)
    u_pre = zeros(Float64, Nx)
    v_pre = zeros(Float64, Nx)
    u_rot = zeros(Float64, Nx)
    v_rot = zeros(Float64, Nx)
    tau_x = zeros(Float64, Nx)
    tau_y = zeros(Float64, Nx)
    angle = zeros(Float64, Nx)
    rcos = zeros(Float64, Nx)
    rsin = zeros(Float64, Nx)
    rnorm = zeros(Float64, Nx)

    for (n, ii) in enumerate(i_indices)
        fi = @inbounds parent(regridder.i)[ii, j_row, 1]
        fj = @inbounds parent(regridder.j)[ii, j_row, 1]
        x_itp = FractionalIndices(fi, fj, nothing)

        u_ext = oc_interpolate(x_itp, time_itp, ua.data, backend, time_indexing)
        v_ext = oc_interpolate(x_itp, time_itp, va.data, backend, time_indexing)
        u_int, v_int = intrinsic_vector(ii, j_row, kN, grid, u_ext, v_ext)
        rot = raw_rotation_terms(ii, j_row)

        u_pre[n] = u_ext
        v_pre[n] = v_ext
        u_rot[n] = u_int
        v_rot[n] = v_int
        tau_x[n] = @inbounds parent(fluxes.x_momentum)[ii, j_row, k]
        tau_y[n] = @inbounds parent(fluxes.y_momentum)[ii, j_row, k]
        angle[n] = rot.θ
        rcos[n] = rot.Rcosθ
        rsin[n] = rot.Rsinθ
        rnorm[n] = rot.R
    end

    return (; u_pre, v_pre, u_rot, v_rot, tau_x, tau_y, angle, rcos, rsin, rnorm)
end

south = row_diagnostics(j_south)
seam  = row_diagnostics(j_seam)
halo  = row_diagnostics(j_halo)

state_u_rows = (; south = logical_row(u_field_atmos, j_south),
                  seam  = logical_row(u_field_atmos, j_seam),
                  halo  = logical_row(u_field_atmos, j_halo),
                  halo_filled = halo_filled_row(u_field_atmos, j_halo))

state_v_rows = (; south = logical_row(v_field_atmos, j_south),
                  seam  = logical_row(v_field_atmos, j_seam),
                  halo  = logical_row(v_field_atmos, j_halo),
                  halo_filled = halo_filled_row(v_field_atmos, j_halo))

flux_x_rows = (; south = logical_row(fluxes.x_momentum, j_south),
                 seam  = logical_row(fluxes.x_momentum, j_seam),
                 halo  = logical_row(fluxes.x_momentum, j_halo),
                 halo_filled = halo_filled_row(fluxes.x_momentum, j_halo))

flux_y_rows = (; south = logical_row(fluxes.y_momentum, j_south),
                 seam  = logical_row(fluxes.y_momentum, j_seam),
                 halo  = logical_row(fluxes.y_momentum, j_halo),
                 halo_filled = halo_filled_row(fluxes.y_momentum, j_halo))

net_tau_u_rows = (; south = logical_row(net_ocean_fluxes.u, j_south),
                    seam  = logical_row(net_ocean_fluxes.u, j_seam),
                    halo  = logical_row(net_ocean_fluxes.u, j_halo),
                    halo_filled = halo_filled_row(net_ocean_fluxes.u, j_halo))

net_tau_v_rows = (; south = logical_row(net_ocean_fluxes.v, j_south),
                    seam  = logical_row(net_ocean_fluxes.v, j_seam),
                    halo  = logical_row(net_ocean_fluxes.v, j_halo),
                    halo_filled = halo_filled_row(net_ocean_fluxes.v, j_halo))

for jj in (j_south, j_seam, j_halo)
    row_name = jj == j_south ? "south" : jj == j_seam ? "seam" : "north_halo"
    for ii in (Hx + Nx - 1, Hx + Nx)
        rot = raw_rotation_terms(ii, jj)
        @info("Raw rotation terms",
              row = row_name,
              ii = ii,
              φpp = rot.φpp,
              φpm = rot.φpm,
              φmp = rot.φmp,
              φmm = rot.φmm,
              Δyp = rot.Δyp,
              Δym = rot.Δym,
              Δxp = rot.Δxp,
              Δxm = rot.Δxm,
              Rcos1 = rot.Rcos1,
              Rcos2 = rot.Rcos2,
              Rcosθ = rot.Rcosθ,
              Rsinθ = rot.Rsinθ,
              R = rot.R,
              θ = rot.θ)
    end
end

@info("Rotation diagnostics",
      seam_nan_u_rot = count(isnan, seam.u_rot),
      seam_nan_v_rot = count(isnan, seam.v_rot),
      seam_nan_angle = count(isnan, seam.angle),
      seam_zero_rnorm = count(x -> x == 0, seam.rnorm),
      halo_nan_u_rot = count(isnan, halo.u_rot),
      halo_nan_v_rot = count(isnan, halo.v_rot),
      halo_nan_angle = count(isnan, halo.angle),
      halo_zero_rnorm = count(x -> x == 0, halo.rnorm),
      south_nan_u_rot = count(isnan, south.u_rot),
      south_nan_v_rot = count(isnan, south.v_rot),
      south_nan_angle = count(isnan, south.angle),
      south_zero_rnorm = count(x -> x == 0, south.rnorm))

@info("Halo synchronization diagnostics",
      state_u_halo_matches_filled = isapprox(state_u_rows.halo, state_u_rows.halo_filled),
      state_v_halo_matches_filled = isapprox(state_v_rows.halo, state_v_rows.halo_filled),
      flux_x_halo_matches_filled = isapprox(flux_x_rows.halo, flux_x_rows.halo_filled),
      flux_y_halo_matches_filled = isapprox(flux_y_rows.halo, flux_y_rows.halo_filled),
      net_tau_u_halo_matches_filled = isapprox(net_tau_u_rows.halo, net_tau_u_rows.halo_filled),
      net_tau_v_halo_matches_filled = isapprox(net_tau_v_rows.halo, net_tau_v_rows.halo_filled),
      state_u_halo_jump = maximum(abs, state_u_rows.halo .- state_u_rows.halo_filled),
      state_v_halo_jump = maximum(abs, state_v_rows.halo .- state_v_rows.halo_filled),
      flux_x_halo_jump = maximum(abs, flux_x_rows.halo .- flux_x_rows.halo_filled),
      flux_y_halo_jump = maximum(abs, flux_y_rows.halo .- flux_y_rows.halo_filled),
      net_tau_u_halo_jump = maximum(abs, net_tau_u_rows.halo .- net_tau_u_rows.halo_filled),
      net_tau_v_halo_jump = maximum(abs, net_tau_v_rows.halo .- net_tau_v_rows.halo_filled))

fig = Figure(size = (1800, 2800))

ax1 = Axis(fig[1, 1], title = "u before rotation", xlabel = "i index", ylabel = "uₑ")
lines!(ax1, south.u_pre, label = "south", linewidth = 3, color = :black)
lines!(ax1, seam.u_pre, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax1, halo.u_pre, label = "north halo", linewidth = 2, color = :dodgerblue)
axislegend(ax1, position = :rb)

ax2 = Axis(fig[1, 2], title = "u after intrinsic_vector", xlabel = "i index", ylabel = "uᵢ")
lines!(ax2, south.u_rot, label = "south", linewidth = 3, color = :black)
lines!(ax2, seam.u_rot, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax2, halo.u_rot, label = "north halo", linewidth = 2, color = :dodgerblue)
axislegend(ax2, position = :rb)

ax3 = Axis(fig[2, 1], title = "v before rotation", xlabel = "i index", ylabel = "vₑ")
lines!(ax3, south.v_pre, label = "south", linewidth = 3, color = :black)
lines!(ax3, seam.v_pre, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax3, halo.v_pre, label = "north halo", linewidth = 2, color = :dodgerblue)
axislegend(ax3, position = :rb)

ax4 = Axis(fig[2, 2], title = "v after intrinsic_vector", xlabel = "i index", ylabel = "vᵢ")
lines!(ax4, south.v_rot, label = "south", linewidth = 3, color = :black)
lines!(ax4, seam.v_rot, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax4, halo.v_rot, label = "north halo", linewidth = 2, color = :dodgerblue)
axislegend(ax4, position = :rb)

ax5 = Axis(fig[3, 1], title = "rotation_angle", xlabel = "i index", ylabel = "θ")
lines!(ax5, south.angle, label = "south", linewidth = 3, color = :black)
lines!(ax5, seam.angle, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax5, halo.angle, label = "north halo", linewidth = 2, color = :dodgerblue)
axislegend(ax5, position = :rb)

ax6 = Axis(fig[3, 2], title = "Rotation norm R", xlabel = "i index", ylabel = "R")
lines!(ax6, south.rnorm, label = "south", linewidth = 3, color = :black)
lines!(ax6, seam.rnorm, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax6, halo.rnorm, label = "north halo", linewidth = 2, color = :dodgerblue)
axislegend(ax6, position = :rb)

ax7 = Axis(fig[4, 1], title = "Rcosθ", xlabel = "i index", ylabel = "Rcosθ")
lines!(ax7, south.rcos, label = "south", linewidth = 3, color = :black)
lines!(ax7, seam.rcos, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax7, halo.rcos, label = "north halo", linewidth = 2, color = :dodgerblue)
axislegend(ax7, position = :rb)

ax8 = Axis(fig[4, 2], title = "Rsinθ", xlabel = "i index", ylabel = "Rsinθ")
lines!(ax8, south.rsin, label = "south", linewidth = 3, color = :black)
lines!(ax8, seam.rsin, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax8, halo.rsin, label = "north halo", linewidth = 2, color = :dodgerblue)
axislegend(ax8, position = :rb)

ax9 = Axis(fig[5, 1], title = "Atmosphere-ocean x-momentum flux", xlabel = "i index", ylabel = "ρτˣ")
lines!(ax9, south.tau_x, label = "south", linewidth = 3, color = :black)
lines!(ax9, seam.tau_x, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax9, halo.tau_x, label = "north halo", linewidth = 2, color = :dodgerblue)
axislegend(ax9, position = :rb)

ax10 = Axis(fig[5, 2], title = "Atmosphere-ocean y-momentum flux", xlabel = "i index", ylabel = "ρτʸ")
lines!(ax10, south.tau_y, label = "south", linewidth = 3, color = :black)
lines!(ax10, seam.tau_y, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax10, halo.tau_y, label = "north halo", linewidth = 2, color = :dodgerblue)
axislegend(ax10, position = :rb)

ax11 = Axis(fig[6, 1], title = "Exchanger state u rows", xlabel = "i index", ylabel = "u")
lines!(ax11, state_u_rows.south, label = "south", linewidth = 3, color = :black)
lines!(ax11, state_u_rows.seam, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax11, state_u_rows.halo, label = "north halo", linewidth = 2, color = :dodgerblue)
lines!(ax11, state_u_rows.halo_filled, label = "north halo after fill!", linewidth = 2, color = :darkgreen, linestyle = :dash)
axislegend(ax11, position = :rb)

ax12 = Axis(fig[6, 2], title = "Exchanger state v rows", xlabel = "i index", ylabel = "v")
lines!(ax12, state_v_rows.south, label = "south", linewidth = 3, color = :black)
lines!(ax12, state_v_rows.seam, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax12, state_v_rows.halo, label = "north halo", linewidth = 2, color = :dodgerblue)
lines!(ax12, state_v_rows.halo_filled, label = "north halo after fill!", linewidth = 2, color = :darkgreen, linestyle = :dash)
axislegend(ax12, position = :rb)

ax13 = Axis(fig[7, 1], title = "Center x-momentum flux rows", xlabel = "i index", ylabel = "ρτˣ")
lines!(ax13, flux_x_rows.south, label = "south", linewidth = 3, color = :black)
lines!(ax13, flux_x_rows.seam, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax13, flux_x_rows.halo, label = "north halo", linewidth = 2, color = :dodgerblue)
lines!(ax13, flux_x_rows.halo_filled, label = "north halo after fill!", linewidth = 2, color = :darkgreen, linestyle = :dash)
axislegend(ax13, position = :rb)

ax14 = Axis(fig[7, 2], title = "Center y-momentum flux rows", xlabel = "i index", ylabel = "ρτʸ")
lines!(ax14, flux_y_rows.south, label = "south", linewidth = 3, color = :black)
lines!(ax14, flux_y_rows.seam, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax14, flux_y_rows.halo, label = "north halo", linewidth = 2, color = :dodgerblue)
lines!(ax14, flux_y_rows.halo_filled, label = "north halo after fill!", linewidth = 2, color = :darkgreen, linestyle = :dash)
axislegend(ax14, position = :rb)

ax15 = Axis(fig[8, 1], title = "Net ocean face stress u rows", xlabel = "i index", ylabel = "τˣ")
lines!(ax15, net_tau_u_rows.south, label = "south", linewidth = 3, color = :black)
lines!(ax15, net_tau_u_rows.seam, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax15, net_tau_u_rows.halo, label = "north halo", linewidth = 2, color = :dodgerblue)
lines!(ax15, net_tau_u_rows.halo_filled, label = "north halo after fill!", linewidth = 2, color = :darkgreen, linestyle = :dash)
axislegend(ax15, position = :rb)

ax16 = Axis(fig[8, 2], title = "Net ocean face stress v rows", xlabel = "i index", ylabel = "τʸ")
lines!(ax16, net_tau_v_rows.south, label = "south", linewidth = 3, color = :black)
lines!(ax16, net_tau_v_rows.seam, label = "seam", linewidth = 3, color = :firebrick)
lines!(ax16, net_tau_v_rows.halo, label = "north halo", linewidth = 2, color = :dodgerblue)
lines!(ax16, net_tau_v_rows.halo_filled, label = "north halo after fill!", linewidth = 2, color = :darkgreen, linestyle = :dash)
axislegend(ax16, position = :rb)

outname = figdir * "jra55_tripolar_rotation_singularity_diagnostics.png"
save(outname, fig)

@info "Saved diagnostic plot" outname
