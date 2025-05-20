using OceanEnsembles
using Oceananigans
using ClimaOcean
using Oceananigans.Fields: location
using JLD2
using Test

function grid_metrics(prefix, ranks)
    file   = jldopen(prefix * "_rank$(ranks[1]).jld2")
    data   = file["serialized/grid"]

    Nx, Ny, Nz = data.Nx, data.Ny*Integer(length(ranks)), data.Nz
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
                data   = file["timeseries/$(fts.name)/$(iter)"][:, :, 1]

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