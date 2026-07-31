using Test
using NumericalEarth
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: tick!, update_state!
using Oceananigans.Units: minutes
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: parent

const Nx = 120
const Ny = 60
const Nz = 4

logical_row(field, j_row, i_indices) = Array(@view parent(field)[i_indices, j_row, 1])

function halo_jump(field, j_halo, i_indices)
    raw = logical_row(field, j_halo, i_indices)
    scratch = deepcopy(field)
    fill_halo_regions!(scratch)
    filled = logical_row(scratch, j_halo, i_indices)
    return maximum(abs, raw .- filled)
end

function forcing_halo_metrics(coupled_model)
    i_indices = 1:Nx
    j_halo = Ny + 1

    exchanger = coupled_model.interfaces.exchanger.atmosphere
    ao_fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes
    net_fluxes = coupled_model.interfaces.net_fluxes.ocean

    return (; state_u = halo_jump(exchanger.state.u, j_halo, i_indices),
              state_v = halo_jump(exchanger.state.v, j_halo, i_indices),
              flux_x = halo_jump(ao_fluxes.x_momentum, j_halo, i_indices),
              flux_y = halo_jump(ao_fluxes.y_momentum, j_halo, i_indices),
              net_u = halo_jump(net_fluxes.u, j_halo, i_indices),
              net_v = halo_jump(net_fluxes.v, j_halo, i_indices))
end

function max_abs_w(model)
    wdata = parent(model.velocities.w)
    return maximum(abs, wdata)
end

function run_cpu_smoke_test(; Δt = 10minutes, updates = 2)
    grid = TripolarGrid(CPU();
                        size = (Nx, Ny, Nz),
                        z = (-20, 0),
                        halo = (7, 7, 7))

    ocean = ocean_simulation(grid;
                             free_surface = SplitExplicitFreeSurface(grid, substeps = 20))

    atmosphere = JRA55PrescribedAtmosphere(CPU(); time_indices_in_memory = 2)
    radiation  = JRA55PrescribedRadiation(CPU(); time_indices_in_memory = 2)

    set!(ocean.model, T = 10, S = 35, u = 0, v = 0)
    fill_halo_regions!((ocean.model.velocities.u, ocean.model.velocities.v, ocean.model.tracers.T, ocean.model.tracers.S))

    coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)

    initial_metrics = forcing_halo_metrics(coupled_model)

    for n in 1:updates
        time_step!(coupled_model.atmosphere, Δt)
        time_step!(coupled_model.radiation, Δt)
        tick!(coupled_model.clock, Δt)
        update_state!(coupled_model)
    end

    final_metrics = forcing_halo_metrics(coupled_model)

    @info "Initial forcing halo metrics" initial_metrics
    @info "Final forcing halo metrics after forcing-only updates" final_metrics updates Δt

    @test all(iszero, Tuple(initial_metrics))
    @test all(iszero, Tuple(final_metrics))

    return (; initial_metrics, final_metrics)
end

run_cpu_smoke_test()
