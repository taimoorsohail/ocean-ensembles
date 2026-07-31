using CairoMakie
using JLD2
using Statistics

with_trailing_slash(path) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator

const DEFAULT_CONDITIONED = joinpath(@__DIR__, "..", "outputs", "RYF_sxtdeg_conditioned_bathymetry.jld2")
const DEFAULT_FIGDIR = with_trailing_slash(expanduser(get(ENV, "FIGDIR", joinpath(@__DIR__, "..", "figures"))))

function usage()
    println("""
    Usage:
      julia --project=ocean-ensembles/analysis ocean-ensembles/analysis/plot_conditioned_bathymetry_diff.jl CONDITIONED_BATHYMETRY.jld2 [REFERENCE_BATHYMETRY.jld2] [OUTPUT_PNG]

    Arguments:
      CONDITIONED_BATHYMETRY.jld2
          Output from NumericalEarth.condition_bathymetry. Expected keys include
          "h" and preferably "original_h".

      REFERENCE_BATHYMETRY.jld2
          Optional external reference bathymetry file. Supported keys: "h",
          "original_h", "bottom_height", or "serialized/grid".
          If omitted, this script uses "original_h" in CONDITIONED_BATHYMETRY.jld2,
          which is the exact bathymetry used as input to conditioning.

      OUTPUT_PNG
          Optional output image path. Defaults to FIGDIR/conditioned_bathymetry_diff.png.
    """)
end

function positive_depth(bathymetry)
    h = Array(bathymetry)
    if any(<(0), h)
        return max.(-h, 0)
    else
        return max.(h, 0)
    end
end

function read_serialized_grid_bathymetry(file)
    haskey(file, "serialized/grid") || return nothing
    grid = file["serialized/grid"]
    hasproperty(grid, :immersed_boundary) || return nothing

    Nx, Ny = grid.Nx, grid.Ny
    Hx, Hy = grid.Hx, grid.Hy
    bottom_height = grid.immersed_boundary.bottom_height

    return positive_depth(bottom_height[Hx+1:Hx+Nx, Hy+1:Hy+Ny, 1])
end

function read_bathymetry(path; prefer_original=false)
    h = jldopen(path, "r") do file
        if prefer_original && haskey(file, "original_h")
            Array(file["original_h"])
        elseif haskey(file, "h")
            Array(file["h"])
        elseif haskey(file, "original_h")
            Array(file["original_h"])
        elseif haskey(file, "bottom_height")
            positive_depth(file["bottom_height"])
        else
            grid_h = read_serialized_grid_bathymetry(file)
            isnothing(grid_h) ? nothing : grid_h
        end
    end

    isnothing(h) || return h

    throw(ArgumentError("Could not find bathymetry in $path. Expected key h, original_h, bottom_height, or serialized/grid."))
end

function read_optional_array(path, key)
    jldopen(path, "r") do file
        return haskey(file, key) ? Array(file[key]) : nothing
    end
end

function warn_if_reference_differs_from_stored_original(conditioned_path, reference_h)
    original_h = read_optional_array(conditioned_path, "original_h")
    isnothing(original_h) && return nothing
    size(original_h) == size(reference_h) || return nothing

    diff = original_h .- reference_h
    max_abs_diff = maximum(abs.(diff))
    max_abs_diff == 0 && return nothing

    @warn "Reference bathymetry differs from original_h stored in conditioned file. The plotted delta will include pre-existing differences between the run bathymetry and the reference, not just conditioning changes." max_abs_diff mean_abs_diff=mean(abs.(diff)) changed_cells=count(!=(0), diff)
    return nothing
end

function mask_land(h)
    plotted = Float64.(h)
    plotted[plotted .<= 0] .= NaN
    return plotted
end

function symmetric_limit(A)
    finite_values = A[isfinite.(A)]
    isempty(finite_values) && return 1.0
    limit = maximum(abs, finite_values)
    return limit == 0 ? 1.0 : limit
end

function depth_limit_for_colorbar(arrays...)
    values = Float64[]
    for A in arrays
        append!(values, A[isfinite.(A)])
    end

    isempty(values) && return 1.0
    limit = maximum(values)
    return limit == 0 ? 1.0 : limit
end

function changed_stats(diff)
    changed = isfinite.(diff) .& (diff .!= 0)
    nchanged = count(changed)
    if nchanged == 0
        return (; changed_cells = 0,
                 max_abs_change = 0.0,
                 mean_abs_change = 0.0,
                 total_change = 0.0)
    end

    Δ = diff[changed]
    return (; changed_cells = nchanged,
             max_abs_change = maximum(abs, Δ),
             mean_abs_change = mean(abs.(Δ)),
             total_change = sum(Δ))
end

function is_figure_path(path)
    ext = lowercase(splitext(path)[2])
    return ext in (".png", ".pdf", ".svg")
end

conditioned_path = length(ARGS) >= 1 ? expanduser(ARGS[1]) : DEFAULT_CONDITIONED

reference_path = if length(ARGS) >= 2 && !is_figure_path(ARGS[2])
    expanduser(ARGS[2])
else
    nothing
end

output_path = if length(ARGS) >= 3
    expanduser(ARGS[3])
elseif length(ARGS) >= 2 && is_figure_path(ARGS[2])
    expanduser(ARGS[2])
else
    joinpath(DEFAULT_FIGDIR, "conditioned_bathymetry_diff.png")
end

if !isfile(conditioned_path)
    usage()
    throw(ArgumentError("Conditioned bathymetry file not found: $conditioned_path"))
end

if !isnothing(reference_path) && !isfile(reference_path)
    usage()
    throw(ArgumentError("Reference bathymetry file not found: $reference_path"))
end

mkpath(dirname(output_path))

@info "Loading conditioned bathymetry" conditioned_path
conditioned_h = read_bathymetry(conditioned_path)

reference_h = if isnothing(reference_path)
    @info "Loading original bathymetry from conditioned file" conditioned_path key="original_h"
    read_bathymetry(conditioned_path; prefer_original=true)
else
    @info "Loading reference bathymetry" reference_path
    read_bathymetry(reference_path)
end

reference_label = isnothing(reference_path) ? "Conditioning input bathymetry" : "External reference bathymetry"

!isnothing(reference_path) && warn_if_reference_differs_from_stored_original(conditioned_path, reference_h)

size(conditioned_h) == size(reference_h) ||
    throw(DimensionMismatch("Conditioned bathymetry size $(size(conditioned_h)) does not match reference size $(size(reference_h))."))

diff = conditioned_h .- reference_h
stats = changed_stats(diff)

reference_plot = mask_land(reference_h)
conditioned_plot = mask_land(conditioned_h)
diff_plot = Float64.(diff)
diff_plot[(reference_h .<= 0) .& (conditioned_h .<= 0)] .= NaN

depth_limit = depth_limit_for_colorbar(reference_plot, conditioned_plot)
diff_limit = symmetric_limit(diff_plot)

fig = Figure(size = (1800, 650))

ax1 = Axis(fig[1, 1], title = reference_label, xlabel = "i", ylabel = "j")
hm1 = heatmap!(ax1, reference_plot; colormap = Reverse(:deep), colorrange = (0, depth_limit), nan_color = :lightgray)
Colorbar(fig[1, 2], hm1, label = "Depth (m)")

ax2 = Axis(fig[1, 3], title = "Conditioned bathymetry", xlabel = "i", ylabel = "j")
hm2 = heatmap!(ax2, conditioned_plot; colormap = Reverse(:deep), colorrange = (0, depth_limit), nan_color = :lightgray)
Colorbar(fig[1, 4], hm2, label = "Depth (m)")

ax3 = Axis(fig[1, 5], title = "Conditioned - reference", xlabel = "i", ylabel = "j")
hm3 = heatmap!(ax3, diff_plot; colormap = :balance, colorrange = (-diff_limit, diff_limit), nan_color = :lightgray)
Colorbar(fig[1, 6], hm3, label = "Depth change (m)")

title = "Bathymetry conditioning difference"
subtitle = "changed cells: $(stats.changed_cells), max |Δh|: $(round(stats.max_abs_change, digits=3)) m, mean |Δh|: $(round(stats.mean_abs_change, digits=3)) m, sum Δh: $(round(stats.total_change, digits=3)) m"
Label(fig[0, 1:6], title, fontsize = 24)
Label(fig[2, 1:6], subtitle, fontsize = 16)

@info "Saving bathymetry comparison figure" output_path changed_cells=stats.changed_cells
save(output_path, fig, px_per_unit = 2)
@info "Saved bathymetry comparison figure" output_path
