module OutputWrangling

using CairoMakie
using Oceananigans
using Oceananigans.Fields: location
using Statistics
using JLD2
using ClimaOcean 

export combine_outputs

function grid_metrics(prefix, ranks)
    file   = jldopen(prefix * "_rank$(ranks[1]).jld2")
    data   = file["serialized/grid"]

    Nx, Ny, Nz = data.Nx, data.Ny, data.Nz
    Hx, Hy, Hz = data.Hx, data.Hy, data.Hz
    nx = Integer(Nx / length(ranks))
    ny = Integer(Ny / length(ranks))

    r_faces = exponential_z_faces(; Nz, depth=5000, h=34)
    z_faces = Oceananigans.MutableVerticalDiscretization(r_faces)
    return Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, z_faces
end

function create_grid(prefix, ranks)
    Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, z_faces = grid_metrics(prefix, ranks)

    grid = TripolarGrid(CPU();
                        size = (Nx, Ny, Nz),
                        z = z_faces,
                        halo = (Hx, Hy, Hz),
                        first_pole_longitude = 70,
                        north_poles_latitude = 55)

    bottom_height = read_bathymetry(prefix, ranks)

    grid  = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))
    return grid
end

function read_bathymetry(prefix, ranks)
    Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, z_faces = grid_metrics(prefix, ranks)

    bottom_height = zeros(Nx, Ny)

    for rank in ranks
        irange = ny * rank + 1 : ny * (rank + 1)
        file   = jldopen(prefix * "_rank$(rank).jld2")
        @show typeof(file["serialized/grid"].immersed_boundary.bottom_height)
        @show [Hx+1:Nx+Hx, Hy+1:ny+Hy,  1]
        data   = file["serialized/grid"].immersed_boundary.bottom_height[Hx+1:Nx+Hx, Hy+1:ny+Hy,  1]
        bottom_height[:, irange] .= data
        close(file)
    end

    return bottom_height
end

function combine_outputs(ranks, prefix, prefix_out)

    grid = create_grid(prefix, ranks)

    file0 = jldopen(prefix * "_rank$(ranks[1]).jld2")
    iters = keys(file0["timeseries/t"])
    times = Float64[file0["timeseries/t/$(iter)"] for iter in iters]
    close(file0)

    utmp = FieldTimeSeries{Face,   Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * ".jld2", name="u")
    vtmp = FieldTimeSeries{Center, Face,   Nothing}(grid, times; backend=OnDisk(), path=prefix_out * ".jld2", name="v")
    Ttmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * ".jld2", name="T")
    Stmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * ".jld2", name="S")
    etmp = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend=OnDisk(), path=prefix_out * ".jld2", name="e")

    function set_distributed_field_time_series!(fts, prefix, ranks)
        Nx, Ny, Nz, Hx, Hy, Hz, nx, ny, z_faces = grid_metrics(prefix, ranks)

        field = Field{location(fts)...}(grid)
        Ny = size(fts, 2)
        for (idx, iter) in enumerate(iters)
            @info "doing iter $idx of $(length(iters))"
            for rank in ranks
                irange = ny * rank + 1 : ny * (rank + 1)
                file   = jldopen(prefix * "_rank$(rank).jld2")
                @show irange, size(field)

                data   = file["timeseries/$(fts.name)/$(iter)"][:, irange, 1]

                interior(field, :, irange, 1) .= data
                close(file)
            end

            set!(fts, field, idx)
        end
    end

    set_distributed_field_time_series!(utmp, prefix_out, ranks)
    set_distributed_field_time_series!(vtmp, prefix_out, ranks)
    set_distributed_field_time_series!(Ttmp, prefix_out, ranks)
    set_distributed_field_time_series!(Stmp, prefix_out, ranks)
    set_distributed_field_time_series!(etmp, prefix_out, ranks)
    return nothing
end
end module
#=
# output_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/outputs/")
# figdir = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/")

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

variables_tracers = ["T", "S"]
variables_velocities =  ["u", "v", "w"]
variables_fluxes = ["latent_heat", "sensible_heat", "water_vapor", "x_momentum", "y_momentum"]
variables_surface = ["T"]#vcat(variables_tracers, variables_velocities)


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

@info "I am loading the fluxes" 
fluxes = create_dict(variables_fluxes, output_path * "fluxes.jld2")
@info "I am loading the surface properties" 
surface = create_dict(variables_surface, output_path * "one_degree_surface_fields.jld2")

fig = Figure(size = (1600, 800))
# axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

time_slice = lastindex(fluxes["latent_heat"].times)

times = fluxes["latent_heat"].times

title = string("Global 1 degree ocean simulation after ",
                         prettytime(times[time_slice] - times[1]))


axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
hm = heatmap!(axs, view(interior(fluxes["latent_heat"][time_slice]), :, :, 1), colorrange = (-500,500), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[1, 2], hm, label = "Latent Heat (W/m2)")

axs = Axis(fig[1, 3], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
hm = heatmap!(axs, view(interior(fluxes["sensible_heat"][time_slice]), :, :, 1), colorrange = (-500,500), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[1, 4], hm, label = "Sensible Heat (W/m2)")
Label(fig[0, :], title)

save(figdir * "surface_fluxes.png", fig, px_per_unit=3)

using Oceananigans.Operators: Az
LH_W = zeros(length(times))  # Ensure the array is 1D with the correct size
SH_W = zeros(length(times))  # Ensure the array is 1D with the correct size

for time_step in eachindex(times)  # Proper way to iterate over indices
    LH_W[time_step] = sum(interior(fluxes["latent_heat"][time_step]*Az))
    SH_W[time_step] = sum(interior(fluxes["sensible_heat"][time_step]*Az))
end

mxtemp = zeros(length(times))  # Ensure the array is 1D with the correct size
mntemp = zeros(length(times))  # Ensure the array is 1D with the correct size
hightmp_counter = zeros(length(times))
for time_step in eachindex(times)  # Proper way to iterate over indices
    mxtemp[time_step] = maximum(surface["T"][time_step])  # Ignore NaNs
    mntemp[time_step] = minimum(surface["T"][time_step])  # Ignore NaNs
    T_array = interior(surface["T"][time_step])
    # Extract the interior data
    interior_data = interior(surface["T"][time_step])

    # Find the (i, j) index of the maximum temperature in the interior
    max_index = argmax(interior_data)
end

fig = Figure(size = (1200, 800))
ax1 = Axis(fig[1, 1], xlabel="Years", ylabel="Latent Heat [W]", title="Latent Heat")
ax2 = Axis(fig[1, 2], xlabel="Years", ylabel="Sensible Heat [W]", title="Sensible Heat")
ax3 = Axis(fig[2,1], xlabel="Years", ylabel="Temperature", title="Surface Temperature Extrema")

lines!(ax1, times/(3600*24*365.25), LH_W, color=:black, label = "ClimaOcean - 1deg Tripolar")
lines!(ax2, times/(3600*24*365.25), SH_W, color=:black)
lines!(ax3, times/(3600*24*365.25), mxtemp, color=:red)
lines!(ax3, times/(3600*24*365.25), mntemp, color=:blue)
axislegend(ax1, position = :rb)

dt_change = 20 / 365.25  # 20 days in years

vlines!(ax1, [dt_change], color=:black, linestyle=:dash, label = "Increasing dt")
vlines!(ax2, [dt_change], color=:black, linestyle=:dash)
vlines!(ax3, [dt_change], color=:black, linestyle=:dash)

save(figdir * "flux_issue.png", fig, px_per_unit=3)

#=
fig = Figure(size = (1200, 800))
# axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

title = string("Global 1 degree ocean simulation after ",
                         prettytime(times[time_slice] - times[1]))

axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
hm = heatmap!(axs, view(interior(surface_props["T"][time_slice]), :, :, 1), colorrange = (-1,31), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[1, 2], hm, label = "Temperature (ᵒC)")

axs = Axis(fig[2, 1], xlabel="Longitude (deg)", ylabel="Depth (m)")
hm = heatmap!(axs, view(interior(surface_props["u"][time_slice]), :, :, 1) , colorrange = (-1,1), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[2, 2], hm, label = "u (m/s)")

axs = Axis(fig[2, 3], xlabel="Longitude (deg)", ylabel="Depth (m)")
hm = heatmap!(axs, view(interior(surface_props["v"][time_slice]), :, :, 1) , colorrange = (-1,1), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[2, 4], hm, label = "v (m/s)")

axs = Axis(fig[1, 3], xlabel="Longitude (deg)", ylabel="Depth (m)")
hm = heatmap!(axs, view(interior(surface_props["S"][time_slice]), :, :, 1) , colorrange = (33, 38), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[1, 4], hm, label = "Salinity (unitless)")

Label(fig[0, :], title)
save(figdir * "surface_fields.png", fig, px_per_unit=3)
=#
=#