using NumericalEarth
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: parent
using Oceananigans.Grids: halo_size
using Oceananigans.Units: minutes
using Oceananigans.TimeSteppers: update_state!, time_step!, tick!
using Dates: DateTime

const data_path = expanduser("/home/tsohail/uom/ocean-ensembles/data/")
const ecco_date = DateTime(1998, 1, 1)
const ecco_dataset = ECCO4Monthly()

const Nx = 120
const Ny = 60
const Nz = 10

interior_i(grid) = (halo_size(grid)[1] + 1):(halo_size(grid)[1] + size(grid, 1))
logical_j(grid, j) = halo_size(grid)[2] + j
logical_k(grid, k) = halo_size(grid)[3] + k
surface_k(grid) = halo_size(grid)[3] + size(grid, 3) + 1
cell_top_k(grid) = halo_size(grid)[3] + size(grid, 3)

function row_values(field, grid, j; k = 1)
    data = parent(field)
    ii = interior_i(grid)
    jj = logical_j(grid, j)
    return Array(@view data[ii, jj, k])
end

function row_stats(field, grid, j; k = 1)
    vals = row_values(field, grid, j; k)
    return (; minimum = minimum(vals), maximum = maximum(vals), maxabs = maximum(abs, vals), mean = sum(vals) / length(vals), finite = all(isfinite, vals))
end

function row_jump(field, grid, j1, j2; k = 1)
    a = row_values(field, grid, j1; k)
    b = row_values(field, grid, j2; k)
    return maximum(abs, a .- b)
end

function report_rows(tag, name, field, grid; k = 1)
    south = row_stats(field, grid, Ny - 1; k)
    seam = row_stats(field, grid, Ny; k)
    halo = row_stats(field, grid, Ny + 1; k)

    @info tag name = name south = south seam = seam halo = halo seam_jump = row_jump(field, grid, Ny - 1, Ny; k) halo_jump = row_jump(field, grid, Ny, Ny + 1; k)
end

function report_state(tag, coupled_model)
    ocean = coupled_model.ocean.model
    grid = ocean.grid

    atmosphere_state = coupled_model.interfaces.exchanger.atmosphere.state
    ao_fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes
    net_ocean_fluxes = coupled_model.interfaces.net_fluxes.ocean

    report_rows(tag, "ocean_w_surface", ocean.velocities.w, grid; k = surface_k(grid))
    report_rows(tag, "ocean_u_top", ocean.velocities.u, grid; k = cell_top_k(grid))
    report_rows(tag, "ocean_v_top", ocean.velocities.v, grid; k = cell_top_k(grid))
    report_rows(tag, "ocean_T_top", ocean.tracers.T, grid; k = cell_top_k(grid))
    report_rows(tag, "ocean_S_top", ocean.tracers.S, grid; k = cell_top_k(grid))

    report_rows(tag, "exchanger_u", atmosphere_state.u, grid; k = 1)
    report_rows(tag, "exchanger_v", atmosphere_state.v, grid; k = 1)
    report_rows(tag, "ao_flux_x", ao_fluxes.x_momentum, grid; k = 1)
    report_rows(tag, "ao_flux_y", ao_fluxes.y_momentum, grid; k = 1)
    report_rows(tag, "net_tau_u", net_ocean_fluxes.u, grid; k = 1)
    report_rows(tag, "net_tau_v", net_ocean_fluxes.v, grid; k = 1)
end

function run_reduced_operational(; Δt = 10minutes)
    grid = TripolarGrid(CPU();
                        size = (Nx, Ny, Nz),
                        z = (-500, 0),
                        halo = (7, 7, 7))

    ocean = ocean_simulation(grid;
                             free_surface = SplitExplicitFreeSurface(grid, substeps = 20))

    set!(ocean.model,
         T = Metadata(:temperature; dates = ecco_date, dataset = ecco_dataset, dir = data_path),
         S = Metadata(:salinity; dates = ecco_date, dataset = ecco_dataset, dir = data_path),
         u = Metadata(:u_velocity; dates = ecco_date, dataset = ecco_dataset, dir = data_path),
         v = Metadata(:v_velocity; dates = ecco_date, dataset = ecco_dataset, dir = data_path))

    fill_halo_regions!((ocean.model.velocities.u, ocean.model.velocities.v,
                        ocean.model.tracers.T, ocean.model.tracers.S))

    atmosphere = JRA55PrescribedAtmosphere(CPU(); time_indices_in_memory = 2)
    radiation  = JRA55PrescribedRadiation(CPU(); time_indices_in_memory = 2)
    coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)

    report_state("Initial reduced operational state", coupled_model)

    time_step!(coupled_model.radiation, Δt)
    report_state("After radiation step", coupled_model)

    time_step!(coupled_model.atmosphere, Δt)
    report_state("After atmosphere step", coupled_model)

    time_step!(coupled_model.ocean, Δt)
    report_state("After ocean step", coupled_model)

    tick!(coupled_model.clock, Δt)
    report_state("After coupled clock tick", coupled_model)

    update_state!(coupled_model)
    report_state("After interface update_state!", coupled_model)
end

run_reduced_operational()
