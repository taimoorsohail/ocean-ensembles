using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2

output_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles/outputs/")
figdir = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles/figures/")

path = output_path * "global_surface_fields.jld2"
var = "T"

T_surface = FieldTimeSeries(path, var)

time_last = Integer(size(T_surface.times)...)

T_lastarray = view(interior(T_surface[time_last]), :, :, 1)
T_lastarray[T_lastarray .== 0.0] .= NaN

maxT, minT = maximum(x for x in T_lastarray if !isnan(x)), minimum(x for x in T_lastarray if !isnan(x))

# unpack grid metrics
Nx = T_surface.grid.underlying_grid.Nx
Ny = T_surface.grid.underlying_grid.Ny

lonv = T_surface.grid.underlying_grid.λᶠᶠᵃ[1:Nx+1, 1:Ny+1]
latv = T_surface.grid.underlying_grid.φᶠᶠᵃ[1:Nx+1, 1:Ny+1]
lonv2 = mod.(lonv .- 250, 360) 

quad_points = [Point2(lonv2[i,j], latv[i,j]) for j in 1:Ny+1, i in 1:Nx+1]
flat_points = reshape(quad_points, :)

# Build quad faces using linear indexing
faces = Face{4}[
    Face{4}((
        (j-1)*(Nx+1) + i,
        (j-1)*(Nx+1) + i + 1,
        j*(Nx+1) + i + 1,
        j*(Nx+1) + i
    )) for j in 1:Ny, i in 1:Nx
]

# Flatten color values (one per cell)
colors = vec(T_lastarray)

# Create figure
fig = Figure()
ax = Axis(fig[1, 1])

# Plot the mesh
mesh!(ax, flat_points, faces;
      color = colors,
      colormap = :coolwarm,
      nan_color = (:gray, 0.0),
      shading = NoShading)

Colorbar(fig[1, 2], colormap=:coolwarm)

display(fig)

# lon = T_surface.grid.underlying_grid.λᶜᶜᵃ[1:Nx, 1:Ny]
# lat = T_surface.grid.underlying_grid.φᶜᶜᵃ[1:Nx, 1:Ny]
#=
polygons = []

for i in 1:Nx-1
    for j in 1:Ny-1
        # Each polygon defined by 4 adjacent points
        lat = [latv[i, j], latv[i+1, j], latv[i+1, j+1], latv[i, j+1]]
        lon = [lonv2[i, j], lonv2[i+1, j], lonv2[i+1, j+1], lonv2[i, j+1]]
        
        # Store the polygon's latitudes and longitudes
        push!(polygons, (lat, lon, T_lastarray[i,j]))
    end
end

# Create the figure and axis
fig = Figure()

# Create an axis in the figure
ax = Axis(fig[1, 1])

# Loop through each polygon and add it to the plot
for (i, (lat, lon, color_value)) in enumerate(polygons)
    # If the color is NaN, choose a default color (e.g., gray)
    if isnan(color_value)
        color = :gray  # Default color for NaNs
    else
        norm_val = (color_value - minT) / (maxT - minT)
        norm_val = clamp(norm_val, 0.0, 1.0)  # Ensure it's within [0,1]
        color = cgrad(:coolwarm)[norm_val]
        end
    
    # # Convert lat/lon coordinates to Point2 objects
    # points = [Point2(lon[k], lat[k]) for k in 1:4]
    
    # Add the polygon to the axis with transparency
    poly!(ax, lon, lat, color=color, strokewidth=0, label="")
    
end

# Show the plot
display(fig)


# longitude_original = field.grid.underlying_grid.λᶜᶜᵃ[1:Nx, 1:Ny]
# lat = field.grid.underlying_grid.φᶜᶜᵃ[1:Nx, 1:Ny]

function plotmap!(ax, array_of_data, field; colorrange, colormap, levels=nothing)#, highclip = automatic, lowclip = automatic)

    
    # # make sure quads are not too distorted
    # lon = mod.(longitude_original .+ -20, 360) .- -20
    # loninsamewindow(l1, l2) = mod(l1 - l2 + 180, 360) + l2 - 180
    # lonv = loninsamewindow.(lonv, reshape(lon, (1, size(lon)...)))

    # create quads
    quad_points = vcat([Point2{Float64}.(lonv[:, i, j], latv[:, i, j]) for i in axes(lonv, 2), j in axes(lonv, 3)]...)
    quad_faces = vcat([begin; j = (i-1) * 4 + 1; [j j+1 j+2; j+2 j+3 j]; end for i in 1:length(quad_points)÷4]...)
    colors_per_point = vcat(fill.(vec(array_of_data), 4)...)

    # create plot
    plt = mesh!(ax, quad_points, quad_faces; color = colors_per_point, shading = NoShading, colormap, colorrange, rasterize = 2)#, highclip, lowclip)
    xlims!(ax, (20, 20 + 360))
    ylims!(ax, (-90, 90))

    # Add contourlines if levels is present
    if !isnothing(levels)
        ilon = sortperm(lon[:,1])
        contour!(ax, lon[ilon, :], lat[ilon, :], x2D[ilon, :]; levels, color=:black, labels = true) # <- looks terrible
        # lon2 = mod.(gridmetrics.lon .- 80, 360) .+ 80
        # contourlevels = Contour.contours(lon2, gridmetrics.lat, x2D, levels)
        # for cl in Contour.levels(contourlevels)
        #     lvl = level(cl) # the z-value of this contour level
        #     for line in Contour.lines(cl)
        #         xs, ys = coordinates(line) # coordinates of this line segment
        #         ls = lines!(ax, xs, ys; color = (:black, 0.5), linewidth = 1)
        #         translate!(ls, 0, 0, 110) # draw the contours above all
        #     end
        # end
    end


    # add coastlines

    cl1 = poly!(ax, land1cut; color = :lightgray, strokecolor = :black, strokewidth = 1)
    translate!(cl1, 0, 0, 100)
    cl2 = poly!(ax, land2cut; color = :lightgray, strokecolor = :black, strokewidth = 1)
    translate!(cl2, 0, 0, 100)

    # move the plot behind the grid so we can see them
    translate!(plt, 0, 0, -100)

    return plt
end

fig = Figure(size=(700, 700))

ax = Axis(fig[1, 1])
colorrange = (0, 1500)
colormap = :viridis

plotmap!(ax,T_lastarray, T_surface; colorrange, colormap)
Colorbar(fig[1,2], plt, label="Ideal mean age (yr)")
# # save plot
# outputfile = joinpath(inputdir, "ideal_mean_age_v2.png")
# @info "Saving ideal mean age as image file:\n  $(outputfile)"
# save(outputfile, fig)

# arch = CPU()

# Nx = 360
# Ny = 180
# Nz = 30

# underlying_grid = TripolarGrid(arch;
#                                size = (Nx, Ny, Nz),
#                                z = (-5000, 0),
#                                halo = (5, 5, 4),
#                                first_pole_longitude = 70,
#                                north_poles_latitude = 55)

# @time bottom_height = regrid_bathymetry(underlying_grid;
#                                         minimum_depth = 10,
#                                         interpolation_passes = 75, # 75 interpolation passes smooth the bathymetry near Florida so that the Gulf Stream is able to flow
#                                         major_basins = 2)

# @time grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

# c = CenterField(grid)
# set!(c, (λ, φ, z) -> (sind(3λ) + 1/3 * sind(5λ)) * cosd(3φ)^2)

# land = interior(c.grid.immersed_boundary.bottom_height) .≥ 0
# land = view(land, :, :, 1)

# c_interior = interior(c, :, :, c.grid.Nz)
# c_interior[land] .= NaN

# cn = c_interior

# fig = Figure(size=(700, 700))

# axc = Axis(fig[1, 1])

# λ, φ, _ = nodes(c)

# # make λ monotonic (250 = 180 + 70 is longitude between poles)
# λ2 = mod.(λ .- 250, 360) .+ 250

# # As a quick workaround, don't plot the last column of λ2, because
# # it is too distorted... Hopefully one day GeoMakie does this automatically
# cf = surface!(axc, λ2[:, 1:end-1], φ[:, 1:end-1], cn[:, 1:end-1]; colorrange=(-1, 1), colormap=:viridis, shading = NoShading, nan_color=:lightgray)

# hidedecorations!(axc)
# hidespines!(axc)

# display(fig)
=#