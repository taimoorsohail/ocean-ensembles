using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.OutputWriters: checkpoint, load_checkpoint_state
using JLD2

include(joinpath(@__DIR__, "RYF_sxtdeg_serial.jl"))

checkpoint_filepath(iteration; dir=output_path, prefix=default_checkpoint_prefix) =
    joinpath(dir, "$(prefix)_iteration$(iteration).jld2")

function validate_checkpoint_file(path)
    jldopen(path, "r") do file
        haskey(file, "simulation/model") || error("Checkpoint file $path does not contain a `simulation/model` entry.")
    end

    return nothing
end

function restore_checkpoint_data!(destination, file, path)
    data_path = path * "/data"

    if haskey(file, data_path)
        parent(destination) .= on_architecture(architecture(destination), file[data_path])
        return nothing
    end

    for name in propertynames(destination)
        child = getproperty(destination, name)
        isnothing(child) && continue

        child_path = path * "/" * String(name)
        if haskey(file, child_path) || haskey(file, child_path * "/data")
            restore_checkpoint_data!(child, file, child_path)
        end
    end

    return nothing
end

function restore_clock!(destination, source)
    destination.time = source.time
    destination.iteration = source.iteration

    if hasproperty(destination, :last_Δt) && hasproperty(source, :last_Δt)
        destination.last_Δt = source.last_Δt
    end

    if hasproperty(destination, :last_stage_Δt) && hasproperty(source, :last_stage_Δt)
        destination.last_stage_Δt = source.last_stage_Δt
    end

    if hasproperty(destination, :stage) && hasproperty(source, :stage)
        destination.stage = source.stage
    end

    return destination
end

function refresh_prescribed_component!(component, source_clock; name)
    isnothing(component) && return nothing

    restore_clock!(component.clock, source_clock)

    try
        time_step!(component, zero(source_clock.time))
        component.clock.iteration -= 1
        component.clock.time = source_clock.time

        if hasproperty(component.clock, :last_Δt) && hasproperty(source_clock, :last_Δt)
            component.clock.last_Δt = source_clock.last_Δt
        end

        if hasproperty(component.clock, :last_stage_Δt) && hasproperty(source_clock, :last_stage_Δt)
            component.clock.last_stage_Δt = source_clock.last_stage_Δt
        end
    catch err
        @warn "Could not refresh prescribed component after manual checkpoint transplant" component=name exception=(err, catch_backtrace())
    end

    return nothing
end

function expose_state_to_repl!(state; source_path=nothing, rewritten_path=nothing)
    Core.eval(Main, :(state = $state))
    Core.eval(Main, :(rewrite_checkpoint_state = $state))
    source_path === nothing || Core.eval(Main, :(rewrite_checkpoint_source_path = $source_path))
    rewritten_path === nothing || Core.eval(Main, :(rewrite_checkpoint_output_path = $rewritten_path))
    return state
end

function transplant_checkpoint_state!(simulation, checkpoint_path)
    checkpoint_state = load_checkpoint_state(checkpoint_path)
    Oceananigans.restore_prognostic_state!(simulation, checkpoint_state)

    if !isnothing(checkpoint_state) && hasproperty(checkpoint_state, :model) && hasproperty(checkpoint_state.model, :clock) && hasproperty(checkpoint_state.model.clock, :last_Δt)
        simulation.Δt = checkpoint_state.model.clock.last_Δt
    end

    return simulation
end

function rewrite_checkpoint_for_current_grid!(arch, run_id;
                                             source_iteration,
                                             source_prefix=default_checkpoint_prefix,
                                             target_prefix=source_prefix * "_current_grid",
                                             state=nothing,
                                             Δt=10minutes,
                                             write_checkpoint=true)
    source_path = checkpoint_filepath(source_iteration; prefix=source_prefix)
    @info "Rewriting checkpoint onto the current grid implementation" source_path target_prefix

    validate_checkpoint_file(source_path)
    source_clock = jldopen(source_path, "r") do file
        file["simulation/model/clock"]
    end

    state = isnothing(state) ?
        build_simulation(arch, run_id; add_outputs=false, Δt, checkpoint_prefix=target_prefix) :
        state

    transplant_checkpoint_state!(state.simulation, source_path)
    expose_state_to_repl!(state; source_path)
    write_checkpoint && Oceananigans.OutputWriters.checkpoint(state.simulation)

    rewritten_path = checkpoint_filepath(source_clock.iteration; prefix=target_prefix)
    @info "Wrote rewritten checkpoint for pickup=true" rewritten_path
    return state, rewritten_path
end

function parse_architecture(arg::AbstractString)
    lowered = lowercase(arg)
    lowered == "cpu" && return CPU()
    lowered == "gpu" && return GPU()
    error("Unknown architecture '$arg'. Use 'CPU' or 'GPU'.")
end

function rewrite_checkpoint(run_id, source_iteration; arch=CPU())

    run_id = run_id
    source_iteration = source_iteration
    arch = arch
    target_prefix = default_checkpoint_prefix * "_current_grid"

    rewritten = rewrite_checkpoint_for_current_grid!(arch, run_id;
                                                     source_iteration=source_iteration,
                                                     target_prefix=target_prefix)

    println("State exposed as Main.state and Main.rewrite_checkpoint_state")
    return rewritten
end

function rewrite_checkpoint!(state; source_iteration,
                             source_prefix=default_checkpoint_prefix,
                             target_prefix=source_prefix * "_current_grid",
                             write_checkpoint=true)
    arch = architecture(state.simulation.model)
    run_id = getproperty(state, :run_id)

    rewritten = rewrite_checkpoint_for_current_grid!(arch, run_id;
                                                     source_iteration=source_iteration,
                                                     source_prefix=source_prefix,
                                                     target_prefix=target_prefix,
                                                     state=state,
                                                     Δt=state.simulation.Δt,
                                                     write_checkpoint=write_checkpoint)

    println("State exposed as Main.state and Main.rewrite_checkpoint_state")
    return rewritten
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
