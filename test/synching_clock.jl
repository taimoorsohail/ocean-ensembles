using ClimaOcean, Oceananigans, Oceananigans.Units

arch = GPU()

for step in 0days:100hours:370days
    @info "step = $(prettytime(step))"
    start_number = Integer(ceil(step/3hours)+1)
    @info "start_number = $(start_number)"
    grid = LatitudeLongitudeGrid(arch, size = (60, 30, 10), z = (-6000, 0), latitude  = (-75, 75), longitude = (0, 360), halo = (6, 6, 3))
    ocean = ocean_simulation(grid; free_surface = SplitExplicitFreeSurface(grid; substeps=10))
    radiation  = Radiation(arch)
    atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(start_number))
    coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)
    
    iteration = 2880
    time = step

    coupled_model.ocean.model.clock.iteration = iteration
    coupled_model.ocean.model.clock.time = time
    coupled_model.atmosphere.clock.iteration = iteration
    coupled_model.atmosphere.clock.time = time
    coupled_model.clock.iteration = iteration
    coupled_model.clock.time = time

    Oceananigans.TimeSteppers.update_state!(coupled_model)
    @info "done update_state!, moving on"
    Oceananigans.TimeSteppers.time_step!(coupled_model, 5minutes)
    @info "done time_step!, moving on"

    iteration = 0
    time = 0.0

    coupled_model.ocean.model.clock.iteration = iteration
    coupled_model.ocean.model.clock.time = time
    coupled_model.atmosphere.clock.iteration = iteration
    coupled_model.atmosphere.clock.time = time
    coupled_model.clock.iteration = iteration
    coupled_model.clock.time = time

    Oceananigans.TimeSteppers.update_state!(coupled_model)
end