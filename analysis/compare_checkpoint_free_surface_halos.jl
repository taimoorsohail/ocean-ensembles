using CairoMakie
using JLD2
using Printf
using Statistics

const DEFAULT_NX = 2160
const DEFAULT_NY = 1080
const DEFAULT_K = 1
const FIELD_CONFIGS = (
    (name = :eta, key = "simulation/model/ocean/model/free_surface/displacement/data", nx_offset = 0, ny_offset = 0),
    (name = :U,   key = "simulation/model/ocean/model/free_surface/barotropic_velocities/U/data", nx_offset = 0, ny_offset = 0),
    (name = :V,   key = "simulation/model/ocean/model/free_surface/barotropic_velocities/V/data", nx_offset = 0, ny_offset = 1),
)

function usage()
    println("""
    Usage:
      julia --project=ocean-ensembles/analysis ocean-ensembles/analysis/compare_checkpoint_free_surface_halos.jl OLD_CHECKPOINT.jld2 NEW_CHECKPOINT.jld2 [Nx] [Ny] [k] [output_prefix]

    Outputs:
      <output_prefix>_interior.png
      <output_prefix>_halos.png
      <output_prefix>_expanded_extra_halos.png
    """)
end

with_trailing_slash(path) = endswith(path, Base.Filesystem.path_separator) ? path : path * Base.Filesystem.path_separator
const DEFAULT_FIGDIR = with_trailing_slash(expanduser(get(ENV, "FIGDIR", joinpath(@__DIR__, "..", "figures"))))

field_interior_dims(config, Nx, Ny) = (Nx + config.nx_offset, Ny + config.ny_offset)

function infer_halos(A, interior_nx, interior_ny)
    sx, sy = size(A, 1), size(A, 2)
    hx_total = sx - interior_nx
    hy_total = sy - interior_ny
    hx_total < 0 && throw(ArgumentError("Array x-size $sx is smaller than interior_nx=$interior_nx."))
    hy_total < 0 && throw(ArgumentError("Array y-size $sy is smaller than interior_ny=$interior_ny."))
    isodd(hx_total) && throw(ArgumentError("Array x-size $sx is incompatible with symmetric halos around interior_nx=$interior_nx."))
    isodd(hy_total) && throw(ArgumentError("Array y-size $sy is incompatible with symmetric halos around interior_ny=$interior_ny."))
    return hx_total ÷ 2, hy_total ÷ 2
end

function read_field(path, key)
    jldopen(path, "r") do file
        haskey(file, key) || throw(KeyError("Could not find key $key in $path"))
        return Array(file[key])
    end
end

function interior_view(A, interior_nx, interior_ny, k)
    hx, hy = infer_halos(A, interior_nx, interior_ny)
    return @view A[hx+1:hx+interior_nx, hy+1:hy+interior_ny, k]
end

function shared_window_views(old_data, new_data, interior_nx, interior_ny, k)
    hx_old, hy_old = infer_halos(old_data, interior_nx, interior_ny)
    hx_new, hy_new = infer_halos(new_data, interior_nx, interior_ny)
    hx_shared = min(hx_old, hx_new)
    hy_shared = min(hy_old, hy_new)

    old_window = @view old_data[hx_old-hx_shared+1:hx_old+interior_nx+hx_shared,
                                hy_old-hy_shared+1:hy_old+interior_ny+hy_shared, k]
    new_window = @view new_data[hx_new-hx_shared+1:hx_new+interior_nx+hx_shared,
                                hy_new-hy_shared+1:hy_new+interior_ny+hy_shared, k]
    return old_window, new_window, hx_shared, hy_shared
end

function halo_only(A, interior_nx, interior_ny, hx, hy)
    B = Float64.(A)
    if hx > 0 || hy > 0
        B[hx+1:hx+interior_nx, hy+1:hy+interior_ny] .= NaN
    end
    return B
end

function expanded_extra_halo_only(new_data, interior_nx, interior_ny, k, hx_shared, hy_shared)
    hx_new, hy_new = infer_halos(new_data, interior_nx, interior_ny)
    B = Float64.(new_data[:, :, k])

    B[hx_new+1:hx_new+interior_nx, hy_new+1:hy_new+interior_ny] .= NaN

    if hx_shared > 0
        B[hx_new-hx_shared+1:hx_new, :] .= NaN
        B[hx_new+interior_nx+1:hx_new+interior_nx+hx_shared, :] .= NaN
    end

    if hy_shared > 0
        B[:, hy_new-hy_shared+1:hy_new] .= NaN
        B[:, hy_new+interior_ny+1:hy_new+interior_ny+hy_shared] .= NaN
    end

    return B
end

function finite_absmax(arrays...)
    maxval = 0.0
    for A in arrays
        vals = A[isfinite.(A)]
        isempty(vals) && continue
        maxval = max(maxval, maximum(abs.(vals)))
    end
    return maxval == 0 ? 1.0 : maxval
end

function field_title(name)
    name == :eta && return "eta"
    name == :U && return "U"
    name == :V && return "V"
    return String(name)
end

function draw_triptych!(fig, row, old_plot, new_plot, diff_plot, label; old_limit=nothing, diff_limit=nothing)
    ax1 = Axis(fig[row, 1], title = "$label old")
    ax2 = Axis(fig[row, 2], title = "$label new")
    ax3 = Axis(fig[row, 3], title = "$label diff")

    hideydecorations!(ax2, grid=false)
    hideydecorations!(ax3, grid=false)

    hm1 = heatmap!(ax1, old_plot; colormap = :viridis,
                   colorrange = isnothing(old_limit) ? automatic : (-old_limit, old_limit))
    hm2 = heatmap!(ax2, new_plot; colormap = :viridis,
                   colorrange = isnothing(old_limit) ? automatic : (-old_limit, old_limit))
    hm3 = heatmap!(ax3, diff_plot; colormap = :balance,
                   colorrange = isnothing(diff_limit) ? automatic : (-diff_limit, diff_limit))

    Colorbar(fig[row, 4], hm1, label = "value")
    Colorbar(fig[row, 5], hm3, label = "new - old")
    return nothing
end

if length(ARGS) < 2
    usage()
    exit(1)
end

old_path = expanduser(ARGS[1])
new_path = expanduser(ARGS[2])
Nx = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : DEFAULT_NX
Ny = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : DEFAULT_NY
k = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : DEFAULT_K
output_prefix = length(ARGS) >= 6 ? expanduser(ARGS[6]) : joinpath(DEFAULT_FIGDIR, "checkpoint_free_surface_compare")

isfile(old_path) || throw(ArgumentError("Old checkpoint file not found: $old_path"))
isfile(new_path) || throw(ArgumentError("New checkpoint file not found: $new_path"))
mkpath(dirname(output_prefix))

interior_fig = Figure(size = (1800, 1200))
halo_fig = Figure(size = (1800, 1200))
extra_halo_fig = Figure(size = (1200, 1200))
Label(interior_fig[0, 1:5], "Free-surface interior comparison (k=$k)", fontsize = 24, font = :bold)
Label(halo_fig[0, 1:5], "Free-surface halo comparison (shared halo overlap, k=$k)", fontsize = 24, font = :bold)
Label(extra_halo_fig[0, 1:2], "Expanded-only halo comparison (non-overlapping halo, k=$k)", fontsize = 24, font = :bold)

for (row, config) in enumerate(FIELD_CONFIGS)
    old_data = read_field(old_path, config.key)
    new_data = read_field(new_path, config.key)
    interior_nx, interior_ny = field_interior_dims(config, Nx, Ny)
    kz = min(size(old_data, 3), size(new_data, 3))
    k_use = min(k, kz)

    old_interior = Float64.(interior_view(old_data, interior_nx, interior_ny, k_use))
    new_interior = Float64.(interior_view(new_data, interior_nx, interior_ny, k_use))
    interior_diff = new_interior .- old_interior

    old_window, new_window, hx_shared, hy_shared = shared_window_views(old_data, new_data, interior_nx, interior_ny, k_use)
    old_halo = halo_only(old_window, interior_nx, interior_ny, hx_shared, hy_shared)
    new_halo = halo_only(new_window, interior_nx, interior_ny, hx_shared, hy_shared)
    halo_diff = new_halo .- old_halo
    expanded_extra_halo = expanded_extra_halo_only(new_data, interior_nx, interior_ny, k_use, hx_shared, hy_shared)

    value_limit = finite_absmax(old_interior, new_interior)
    diff_limit = finite_absmax(interior_diff)
    halo_value_limit = finite_absmax(old_halo, new_halo)
    halo_diff_limit = finite_absmax(halo_diff)
    extra_halo_limit = finite_absmax(expanded_extra_halo)

    label = field_title(config.name)
    draw_triptych!(interior_fig, row, old_interior, new_interior, interior_diff, "$label interior";
                   old_limit=value_limit, diff_limit=diff_limit)
    draw_triptych!(halo_fig, row, old_halo, new_halo, halo_diff, "$label halo";
                   old_limit=halo_value_limit, diff_limit=halo_diff_limit)

    ax_extra = Axis(extra_halo_fig[row, 1], title = "$label expanded-only halo")
    hm_extra = heatmap!(ax_extra, expanded_extra_halo; colormap = :viridis,
                        colorrange = (-extra_halo_limit, extra_halo_limit))
    Colorbar(extra_halo_fig[row, 2], hm_extra, label = "value")

    @info "Prepared comparison plots" field=config.name old_size=size(old_data) new_size=size(new_data) interior_size=(interior_nx, interior_ny) k=k_use shared_halo=(hx_shared, hy_shared)
end

interior_path = output_prefix * "_interior.png"
halo_path = output_prefix * "_halos.png"
extra_halo_path = output_prefix * "_expanded_extra_halos.png"
save(interior_path, interior_fig, px_per_unit=2)
save(halo_path, halo_fig, px_per_unit=2)
save(extra_halo_path, extra_halo_fig, px_per_unit=2)

@info "Saved checkpoint comparison figures" interior_path halo_path extra_halo_path
