using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2

# output_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/outputs/")
# figdir = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/")

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

OTC_file = "ocean_tracer_content_RYF1deg.jld2"
masstrans_file = "mass_transport_RYF1deg.jld2"
suffix = "_skintemp"

variables_basins = ["_global_", "_atl_", "_ipac_"]
variables_diags = ["zonal", "depth", "tot"]
variables_tracers = ["T", "S", "dV"]
variables_velocities =  ["u", "v", "w"]
variables_vel_volumes = ["dV_u", "dV_v", "dV_w"]

tracercontent_vars = vec([tracer * basin * diag for tracer in variables_tracers,
                         basin in variables_basins,
                         diag in variables_diags])

transport_vars = vcat(
    vec([velocity * basin * diag for velocity in variables_velocities,
                                 basin in variables_basins,
                                 diag in variables_diags]),
    vec([volume * basin * diag for volume in variables_vel_volumes,
                               basin in variables_basins,
                               diag in variables_diags])
)

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

@info "I am loading the otc" 
OTC = create_dict(tracercontent_vars, output_path * OTC_file)

@info "I am loading the mass transport" 
masstrans = create_dict(transport_vars, output_path * masstrans_file)

# Define the order you want for tracers and basins
tracers = ["T", "S", "dV"]
velocities = ["u", "v", "w"]
basins = ["global", "atl", "ipac"]
minv = [-1,34,1e6]
maxv = [31,38,1e18]
end_time = lastindex(masstrans["u_ipac_zonal"].times)

function OTC_visualisation(var, tracers, basins; minv, maxv, diagnostic = "integrate", space = "zonal", dir = "./", start_time = 1, end_time = 2, suffix= "")
    fig = Figure(size = (1600, 800))

    # Loop through and place plots accordingly
    for (row, basin) in enumerate(basins)
        for (col, tracer) in enumerate(tracers)
            if occursin("T", tracer)
                rho0 = 1025
                cp = 4000
                multiplier = rho0*cp
            else
                multiplier = 1
            end
            key_dv = "dV_$(basin)_$(space)"
            key = "$(tracer)_$(basin)_$(space)"  # Compose the key
            if space == "zonal"
                ax = Axis(fig[row, 2col-1], title=String(key))
            else
                ax = Axis(fig[row, col], title=String(key))
            end

            if start_time == "last"
                time_slice = lastindex(var[key].times)
            else
                time_slice = start_time:end_time
            end
            
            if diagnostic == "integrate"
                data = view(var[key].*multiplier, :, :, 1, time_slice)
            elseif diagnostic == "average"
                data = view(var[key], :, :, 1, time_slice)./view(var[key_dv], :, :, 1, time_slice)
            end

            if space == "tot"
                if diagnostic == "integrate"
                    data = interior(var[key]).*multiplier
                elseif diagnostic == "average"
                    data = interior(var[key])./(var[key_dv])
                end
            end

            if space == "zonal"
                heatmap!(ax, view(data, 1, :, :), colormap=:viridis, colorrange = (minv[col], maxv[col]))
                Colorbar(fig[row, 2col], label="Units", width=15)
            elseif space == "depth"
                lines!(ax, view(data, 1, 1, :))
            elseif space == "tot"
                lines!(ax, view(data, 1, 1, 1, :))
            else
                throw(ArgumentError("space must be one of zonal, depth or total"))
            end

        end
    end

    title = string("Global 1 degree ocean simulation after ",
                            prettytime(times[time_slice[2]] - times[time_slice[1]]))

    fig[0, :] = Label(fig, "Global 1° ocean simulation after $(prettytime(times[time_slice[2]] - times[time_slice[1]]))", fontsize=24)

    save(dir*"$(space)_$(diagnostic)_tracer_content" * suffix *".png", fig, px_per_unit=3)
end 

OTC_visualisation(OTC, tracers, basins; minv, maxv, diagnostic = "integrate", space = "zonal", dir = figdir, start_time = end_time-75, end_time = end_time, suffix = suffix)
OTC_visualisation(OTC, tracers, basins; minv, maxv, diagnostic = "integrate", space = "depth", dir = figdir, start_time = end_time-75, end_time = end_time, suffix = suffix)
OTC_visualisation(OTC, tracers, basins; minv, maxv, diagnostic = "integrate", space = "tot", dir = figdir, start_time = end_time-75, end_time = end_time, suffix = suffix)
OTC_visualisation(OTC, tracers, basins; minv, maxv, diagnostic = "average", space = "zonal", dir = figdir, start_time = end_time-75, end_time = end_time, suffix = suffix)
OTC_visualisation(OTC, tracers, basins; minv, maxv, diagnostic = "average", space = "depth", dir = figdir, start_time = end_time-75, end_time = end_time, suffix = suffix)
OTC_visualisation(OTC, tracers, basins; minv, maxv, diagnostic = "average", space = "tot", dir = figdir, start_time = end_time-75, end_time = end_time, suffix = suffix)

function masstrans_visualisation(var, tracers, basins; minv, maxv, diagnostic = "integrate", space = "zonal", dir = "./", start_time = 1, end_time = 2, suffix= "")
    fig = Figure(size = (1600, 800))

    # Loop through and place plots accordingly
    for (row, basin) in enumerate(basins)
        for (col, tracer) in enumerate(tracers)
            key_dv = "dV_$(tracer)_$(basin)_$(space)"
            key = "$(tracer)_$(basin)_$(space)"  # Compose the key
            if space == "zonal"
                ax = Axis(fig[row, 2col-1], title=String(key))
            else
                ax = Axis(fig[row, col], title=String(key))
            end

            if start_time == "last"
                time_slice = lastindex(var[key].times)
            else
                time_slice = start_time:end_time
            end
            
            if diagnostic == "integrate"
                data = view(var[key], :, :, 1, time_slice)
            elseif diagnostic == "average"
                data = view(var[key], :, :, 1, time_slice)./view(var[key_dv], :, :, 1, time_slice)
            end

            if space == "tot"
                if diagnostic == "integrate"
                    data = interior(var[key])
                elseif diagnostic == "average"
                    data = interior(var[key])./(var[key_dv])
                end
            end

            if space == "zonal"
                heatmap!(ax, view(data, 1, :, :), colormap=:viridis, colorrange = (minv[col], maxv[col]))
                Colorbar(fig[row, 2col], label="Units", width=15)
            elseif space == "depth"
                lines!(ax, view(data, 1, 1, :))
            elseif space == "tot"
                lines!(ax, view(data, 1, 1, 1, :))
            else
                throw(ArgumentError("space must be one of zonal, depth or total"))
            end

        end
    end

    title = string("Global 1 degree ocean simulation after ",
                            prettytime(times[time_slice[2]] - times[time_slice[1]]))

    fig[0, :] = Label(fig, "Global 1° ocean simulation after $(prettytime(times[time_slice[2]] - times[time_slice[1]]))", fontsize=24)

    save(dir*"$(space)_$(diagnostic)_voltrans" * suffix * ".png", fig, px_per_unit=3)
end

masstrans_visualisation(masstrans, velocities, basins; minv, maxv, diagnostic = "integrate", space = "zonal", dir = figdir, start_time = end_time-75, end_time = end_time, suffix = suffix)
masstrans_visualisation(masstrans, velocities, basins; minv, maxv, diagnostic = "integrate", space = "depth", dir = figdir, start_time = end_time-75, end_time = end_time, suffix = suffix)
masstrans_visualisation(masstrans, velocities, basins; minv, maxv, diagnostic = "integrate", space = "tot", dir = figdir, start_time = end_time-75, end_time = end_time, suffix = suffix)
masstrans_visualisation(masstrans, velocities, basins; minv, maxv, diagnostic = "average", space = "zonal", dir = figdir, start_time = end_time-75, end_time = end_time, suffix = suffix)
masstrans_visualisation(masstrans, velocities, basins; minv, maxv, diagnostic = "average", space = "depth", dir = figdir, start_time = end_time-75, end_time = end_time, suffix = suffix)
masstrans_visualisation(masstrans, velocities, basins; minv, maxv, diagnostic = "average", space = "tot", dir = figdir, start_time = end_time-75, end_time = end_time, suffix = suffix)