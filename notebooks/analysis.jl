using CairoMakie, GLMakie
using Oceananigans  # From local
using Statistics
using JLD2

experiment_path = expanduser("/g/data/v46/txs156/ocean-ensembles/")

variables_basins = ["_global", "_atlantic", "_pacific"]
variables_tracers = ["T", "S", "u", "v", "w", "dV"]

variables = [tracer * basin for tracer in variables_tracers, basin in variables_basins]
variables = vec(variables)  # Flatten into a 1D list

println(variables)

int_za = Dict()
surface_props = Dict()

for var in variables_tracers
    try
        # Surface
        surface_props[var] = FieldTimeSeries(experiment_path * "global_surface_fields.jld2", var)
    catch e
        if e isa KeyError
            @warn "Skipping variable $var: Key not found in file."
        else
            rethrow(e)
        end
    end
end

for var in variables
    try
        # # Averages
        # avg[var] = FieldTimeSeries(experiment_path * "averaged_data.jld2", var)
        # avg_za[var] = FieldTimeSeries(experiment_path * "zonal_averaged_data.jld2", var)
        # avg_depth[var] = FieldTimeSeries(experiment_path * "depth_averaged_data.jld2", var)
        # Integrals
        int_za[var] = FieldTimeSeries(experiment_path * "zonally_integrated_data.jld2", var)
        # int_depth[var] = FieldTimeSeries(experiment_path * "depth_integrated_data.jld2", var)
    catch e
        if e isa KeyError
            @warn "Skipping variable $var: Key not found in file."
        else
            rethrow(e)
        end
    end
end
