using OceanEnsembles
using Oceananigans
using ClimaOcean
using Oceananigans.Fields: location
using JLD2
using Test

prefix = "/g/data/v46/txs156/ocean-ensembles/outputs/global_surface_fields_distributedGPU"
prefix_out = "/g/data/v46/txs156/ocean-ensembles/outputs/global_surface_fields_distributedGPU"
ranks = 0:1

if isfile(prefix_out * ".jld2")
    @info "File found! Deleting..."
    rm(prefix_out * ".jld2")
    combine_outputs(ranks, prefix, prefix_out)
    @test isfile(prefix_out * ".jld2")
else
    combine_outputs(ranks, prefix, prefix_out)
    @test isfile(prefix_out * ".jld2")
end

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")
filename_surf = "global_surface_fields_distributedGPU.jld2"

variables_tracers = ["T", "S"]
variables_velocities =  ["u", "v", "w"]
variables_surface = vcat(variables_tracers, variables_velocities)

function create_dict(vars, path)
    dicts = Dict()
    for var in vars
        try
            # Surface
            dicts[var] = FieldTimeSeries(path, var)
        catch e
            if e isa KeyError
                @warn "Skipping variable $var: Key not found in file."
            else
                rethrow(e)
            end
        end
    end
    return dicts
end

@info "I am loading the surface properties" 
surface = create_dict(variables_surface, output_path * filename_surf)
times = surface["T"].times
time_slice = lastindex(surface["T"].times)
rewind = 3
###### SURFACE FIELDS ######

fig = Figure(size = (1200, 800))
# axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

title = string("Global 1 degree ocean simulation after ",
                         prettytime(times[time_slice-rewind] - times[1]))

axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
hm = heatmap!(axs, view(interior(surface["T"][time_slice-rewind]), :, :, 1), colorrange = (-1,31), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[1, 2], hm, label = "Temperature (ᵒC)")

axs = Axis(fig[2, 1], xlabel="Longitude (deg)", ylabel="Depth (m)")
hm = heatmap!(axs, view(interior(surface["u"][time_slice-rewind]), :, :, 1) , colorrange = (-1,1), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[2, 2], hm, label = "u (m/s)")

axs = Axis(fig[2, 3], xlabel="Longitude (deg)", ylabel="Depth (m)")
hm = heatmap!(axs, view(interior(surface["v"][time_slice-rewind]), :, :, 1) , colorrange = (-1,1), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[2, 4], hm, label = "v (m/s)")

axs = Axis(fig[1, 3], xlabel="Longitude (deg)", ylabel="Depth (m)")
hm = heatmap!(axs, view(interior(surface["S"][time_slice-rewind]), :, :, 1) , colorrange = (33, 38), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[1, 4], hm, label = "Salinity (unitless)")

Label(fig[0, :], title)
save(figdir * "surface_fields.png", fig, px_per_unit=3)