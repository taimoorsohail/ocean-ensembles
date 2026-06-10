using CairoMakie
using NumericalEarth
using Oceananigans
using Oceananigans.Fields: FractionalIndices
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

ua = atmosphere.velocities.u
va = atmosphere.velocities.v
times = ua.times
time_indexing = ua.time_indexing
backend = ua.backend
t = coupled_model.clock.time
time_itp = cpu_interpolating_time_indices(CPU(), times, time_indexing, t)

Hx, Hy, Hz = halo_size(grid)
i_indices = 1:Nx
j_south = Ny - 1
j_seam  = Ny
k = 1
kN = size(grid, 3)

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
        fi = @inbounds regridder.i[ii, j_row, 1]
        fj = @inbounds regridder.j[ii, j_row, 1]
        x_itp = FractionalIndices(fi, fj, nothing)

        u_ext = oc_interpolate(x_itp, time_itp, ua.data, backend, time_indexing)
        v_ext = oc_interpolate(x_itp, time_itp, va.data, backend, time_indexing)
        u_int, v_int = intrinsic_vector(ii, j_row, kN, grid, u_ext, v_ext)
        rot = raw_rotation_terms(ii, j_row)

        u_pre[n] = u_ext
        v_pre[n] = v_ext
        u_rot[n] = u_int
        v_rot[n] = v_int
        tau_x[n] = @inbounds fluxes.x_momentum[ii, j_row, k]
        tau_y[n] = @inbounds fluxes.y_momentum[ii, j_row, k]
        angle[n] = rot.θ
        rcos[n] = rot.Rcosθ
        rsin[n] = rot.Rsinθ
        rnorm[n] = rot.R
    end

    return (; u_pre, v_pre, u_rot, v_rot, tau_x, tau_y, angle, rcos, rsin, rnorm)
end

south = row_diagnostics(j_south)
seam  = row_diagnostics(j_seam)

for jj in (j_south, j_seam)
    row_name = jj == j_south ? "south" : "seam"
    for ii in (Nx - 1, Nx)
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
      south_nan_u_rot = count(isnan, south.u_rot),
      south_nan_v_rot = count(isnan, south.v_rot),
      south_nan_angle = count(isnan, south.angle),
      south_zero_rnorm = count(x -> x == 0, south.rnorm))

fig = Figure(size = (1600, 1800))

ax1 = Axis(fig[1, 1], title = "u before rotation", xlabel = "i index", ylabel = "uₑ")
lines!(ax1, south.u_pre, label = "south", linewidth = 3, color = :black)
lines!(ax1, seam.u_pre, label = "seam", linewidth = 3, color = :firebrick)
axislegend(ax1, position = :rb)

ax2 = Axis(fig[1, 2], title = "u after intrinsic_vector", xlabel = "i index", ylabel = "uᵢ")
lines!(ax2, south.u_rot, label = "south", linewidth = 3, color = :black)
lines!(ax2, seam.u_rot, label = "seam", linewidth = 3, color = :firebrick)
axislegend(ax2, position = :rb)

ax3 = Axis(fig[2, 1], title = "v before rotation", xlabel = "i index", ylabel = "vₑ")
lines!(ax3, south.v_pre, label = "south", linewidth = 3, color = :black)
lines!(ax3, seam.v_pre, label = "seam", linewidth = 3, color = :firebrick)
axislegend(ax3, position = :rb)

ax4 = Axis(fig[2, 2], title = "v after intrinsic_vector", xlabel = "i index", ylabel = "vᵢ")
lines!(ax4, south.v_rot, label = "south", linewidth = 3, color = :black)
lines!(ax4, seam.v_rot, label = "seam", linewidth = 3, color = :firebrick)
axislegend(ax4, position = :rb)

ax5 = Axis(fig[3, 1], title = "rotation_angle", xlabel = "i index", ylabel = "θ")
lines!(ax5, south.angle, label = "south", linewidth = 3, color = :black)
lines!(ax5, seam.angle, label = "seam", linewidth = 3, color = :firebrick)
axislegend(ax5, position = :rb)

ax6 = Axis(fig[3, 2], title = "Rotation norm R", xlabel = "i index", ylabel = "R")
lines!(ax6, south.rnorm, label = "south", linewidth = 3, color = :black)
lines!(ax6, seam.rnorm, label = "seam", linewidth = 3, color = :firebrick)
axislegend(ax6, position = :rb)

ax7 = Axis(fig[4, 1], title = "Rcosθ", xlabel = "i index", ylabel = "Rcosθ")
lines!(ax7, south.rcos, label = "south", linewidth = 3, color = :black)
lines!(ax7, seam.rcos, label = "seam", linewidth = 3, color = :firebrick)
axislegend(ax7, position = :rb)

ax8 = Axis(fig[4, 2], title = "Rsinθ", xlabel = "i index", ylabel = "Rsinθ")
lines!(ax8, south.rsin, label = "south", linewidth = 3, color = :black)
lines!(ax8, seam.rsin, label = "seam", linewidth = 3, color = :firebrick)
axislegend(ax8, position = :rb)

ax9 = Axis(fig[5, 1], title = "Atmosphere-ocean x-momentum flux", xlabel = "i index", ylabel = "ρτˣ")
lines!(ax9, south.tau_x, label = "south", linewidth = 3, color = :black)
lines!(ax9, seam.tau_x, label = "seam", linewidth = 3, color = :firebrick)
axislegend(ax9, position = :rb)

ax10 = Axis(fig[5, 2], title = "Atmosphere-ocean y-momentum flux", xlabel = "i index", ylabel = "ρτʸ")
lines!(ax10, south.tau_y, label = "south", linewidth = 3, color = :black)
lines!(ax10, seam.tau_y, label = "seam", linewidth = 3, color = :firebrick)
axislegend(ax10, position = :rb)

outname = figdir * "jra55_tripolar_rotation_singularity_diagnostics.png"
save(outname, fig)

@info "Saved diagnostic plot" outname
