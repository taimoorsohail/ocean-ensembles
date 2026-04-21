using CairoMakie
using NumericalEarth
using NumericalEarth.ORCA
using Oceananigans
using Oceananigans.Fields: CenterField, instantiate, location

const DATA_PATH = expanduser("/g/data/v46/txs156/ocean-ensembles-2/data/")
const FIGDIR = expanduser("/g/data/v46/txs156/ocean-ensembles-2/figures/")

# Match RYF_sxtdeg.jl TripolarGrid setup.
const NX = Int(360 * 12)
const NY = Int(180 * 12)
const NZ = Int(75)
const HALO = (7, 7, 7)
const ORCA_HALO = HALO
const DEPTH = -5500.0
const ORCA12_SOUTH_ROWS_TO_REMOVE = 43

# Plot density controls (reduce visual clutter and runtime cost).
const ORCA_ROW_STRIDE = 96
const ORCA_COL_STRIDE = 96
const ORCA_LAND_STRIDE = 10
const TRIPOLAR_ROW_STRIDE = 96
const TRIPOLAR_COL_STRIDE = 96

@inline function wrap_longitudes(lon)
    return mod.(Float32.(lon) .+ 180f0, 360f0) .- 180f0
end

function to_degrees_if_needed(lon, lat)
    max_abs_lon = maximum(abs, lon)
    max_abs_lat = maximum(abs, lat)
    if max_abs_lon <= 4.0 && max_abs_lat <= 2.0
        return Float32.(rad2deg.(lon)), Float32.(rad2deg.(lat))
    end
    return Float32.(lon), Float32.(lat)
end

function orca_dataset_and_rows_to_remove()
    orca_mod = NumericalEarth.ORCA

    if isdefined(orca_mod, :ORCA12)
        dataset_ctor = getfield(orca_mod, :ORCA12)
        return dataset_ctor(), ORCA12_SOUTH_ROWS_TO_REMOVE
    end

    error("Could not find ORCA12() in NumericalEarth.ORCA.")
end

function center_lon_lat(grid)
    center_field = CenterField(grid)
    loc = instantiate.(location(center_field))
    base_grid = hasproperty(grid, :underlying_grid) ? grid.underlying_grid : grid

    λnodes_fn = if isdefined(Oceananigans.OrthogonalSphericalShellGrids, :λnodes)
        getfield(Oceananigans.OrthogonalSphericalShellGrids, :λnodes)
    elseif isdefined(Oceananigans.Grids, :λnodes)
        getfield(Oceananigans.Grids, :λnodes)
    else
        error("Could not resolve λnodes in Oceananigans.")
    end

    φnodes_fn = if isdefined(Oceananigans.OrthogonalSphericalShellGrids, :φnodes)
        getfield(Oceananigans.OrthogonalSphericalShellGrids, :φnodes)
    elseif isdefined(Oceananigans.Grids, :φnodes)
        getfield(Oceananigans.Grids, :φnodes)
    else
        error("Could not resolve φnodes in Oceananigans.")
    end

    lon_raw = Array(λnodes_fn(base_grid, loc..., with_halos = false))
    lat_raw = Array(φnodes_fn(base_grid, loc..., with_halos = false))

    ndims(lon_raw) == 3 && (lon_raw = lon_raw[:, :, 1])
    ndims(lat_raw) == 3 && (lat_raw = lat_raw[:, :, 1])

    lon, lat = to_degrees_if_needed(lon_raw, lat_raw)
    lon = wrap_longitudes(lon)
    lat = Float32.(lat)

    return lon, lat
end

function land_mask_from_grid(grid, expected_size)
    if !hasproperty(grid, :immersed_boundary) || !hasproperty(grid.immersed_boundary, :bottom_height)
        @warn "ORCAGrid has no immersed bottom_height; plotting grid lines only."
        return falses(expected_size)
    end

    bottom_height = try
        Array(interior(grid.immersed_boundary.bottom_height))
    catch
        Array(grid.immersed_boundary.bottom_height)
    end

    ndims(bottom_height) == 3 && (bottom_height = bottom_height[:, :, 1])

    if size(bottom_height) != expected_size
        @warn "Land-mask size does not match lon/lat size; plotting without land points." bottom_height_size = size(bottom_height) expected_size
        return falses(expected_size)
    end

    land = (.!isfinite.(bottom_height)) .| (bottom_height .>= -1f-6)
    return BitMatrix(land)
end

function build_orca12_lon_lat_land()
    dataset, south_rows_to_remove = orca_dataset_and_rows_to_remove()
    z_faces = ExponentialDiscretization(NZ, DEPTH, 0, mutable = true)

    grid = ORCAGrid(CPU();
                    dataset,
                    Nz = NZ,
                    z = z_faces,
                    halo = HALO,
                    south_rows_to_remove,
                    dir = DATA_PATH)

    lon, lat = center_lon_lat(grid)
    land = land_mask_from_grid(grid, size(lon))

    return lon, lat, land, dataset
end

function build_tripolar_grid()
    z_faces = ExponentialDiscretization(NZ, DEPTH, 0, mutable = true)
    return TripolarGrid(CPU(); size = (NX, NY, NZ), z = z_faces, halo = HALO)
end

function sample_land_points(lon, lat, land; stride = ORCA_LAND_STRIDE)
    xs = Float32[]
    ys = Float32[]
    nrows, ncols = size(land)

    for row in 1:stride:nrows, col in 1:stride:ncols
        if land[row, col]
            x = lon[row, col]
            y = lat[row, col]
            if isfinite(x) && isfinite(y)
                push!(xs, x)
                push!(ys, y)
            end
        end
    end

    return xs, ys
end

function draw_segmented_line!(ax, xs, ys; color = :black, linewidth = 0.35f0, lon_jump = 140f0, lat_jump = 25f0)
    n = length(xs)
    n < 2 && return

    start_idx = 1
    for k in 2:n
        prev_ok = isfinite(xs[k - 1]) && isfinite(ys[k - 1])
        curr_ok = isfinite(xs[k]) && isfinite(ys[k])
        split = !prev_ok || !curr_ok || abs(xs[k] - xs[k - 1]) > lon_jump || abs(ys[k] - ys[k - 1]) > lat_jump

        if split
            if k - start_idx >= 2
                lines!(ax, @view(xs[start_idx:k-1]), @view(ys[start_idx:k-1]); color, linewidth)
            end
            start_idx = k
        end
    end

    if n - start_idx + 1 >= 2
        lines!(ax, @view(xs[start_idx:n]), @view(ys[start_idx:n]); color, linewidth)
    end
end

function draw_curvilinear_grid!(ax, lon, lat; row_stride, col_stride, color = :black, linewidth = 0.35f0)
    nrows, ncols = size(lon)

    rows = collect(1:row_stride:nrows)
    rows[end] != nrows && push!(rows, nrows)

    cols = collect(1:col_stride:ncols)
    cols[end] != ncols && push!(cols, ncols)

    for row in rows
        draw_segmented_line!(ax, vec(@view(lon[row, :])), vec(@view(lat[row, :])); color, linewidth)
    end

    for col in cols
        draw_segmented_line!(ax, vec(@view(lon[:, col])), vec(@view(lat[:, col])); color, linewidth)
    end
end

function make_comparison_figure(orca_lon, orca_lat, orca_land, tripolar_lon, tripolar_lat; dataset_name::String)
    fig = Figure(size = (2200, 980), fontsize = 18)

    ax_orca = Axis(fig[1, 1];
                   title = "ORCA12 Grid + Land",
                   xlabel = "Longitude (deg)",
                   ylabel = "Latitude (deg)",
                   aspect = DataAspect())

    ax_tripolar = Axis(fig[1, 2];
                       title = "TripolarGrid Output (Nx=$NX, Ny=$NY, Nz=$NZ, halo=$HALO)",
                       xlabel = "Longitude (deg)",
                       ylabel = "Latitude (deg)",
                       aspect = DataAspect())

    land_x, land_y = sample_land_points(orca_lon, orca_lat, orca_land)
    scatter!(ax_orca, land_x, land_y; markersize = 1.5, color = (:gray60, 0.85))
    draw_curvilinear_grid!(ax_orca, orca_lon, orca_lat;
                           row_stride = ORCA_ROW_STRIDE,
                           col_stride = ORCA_COL_STRIDE,
                           color = (:black, 0.42),
                           linewidth = 0.35)

    draw_curvilinear_grid!(ax_tripolar, tripolar_lon, tripolar_lat;
                           row_stride = TRIPOLAR_ROW_STRIDE,
                           col_stride = TRIPOLAR_COL_STRIDE,
                           color = (:dodgerblue4, 0.6),
                           linewidth = 0.45)

    for ax in (ax_orca, ax_tripolar)
        xlims!(ax, -180, 180)
        ylims!(ax, -90, 90)
    end

    Label(fig[0, 1:2],
          "ORCA source: ORCAGrid(dataset=$dataset_name, halo=$ORCA_HALO)\nTripolar setup copied from RYF_sxtdeg.jl: size=($NX, $NY, $NZ), halo=$HALO, depth=$DEPTH m",
          tellwidth = false,
          fontsize = 15)

    return fig
end

function main()
    @info "Building ORCAGrid and extracting land mask..."
    orca_lon, orca_lat, orca_land, dataset = build_orca12_lon_lat_land()

    @info "Building TripolarGrid coordinates..." NX NY NZ HALO DEPTH
    tripolar_grid = build_tripolar_grid()
    tripolar_lon, tripolar_lat = center_lon_lat(tripolar_grid)

    @info "Creating comparison figure..."
    fig = make_comparison_figure(orca_lon, orca_lat, orca_land, tripolar_lon, tripolar_lat;
                                 dataset_name = string(typeof(dataset)))

    mkpath(FIGDIR)
    outfile = joinpath(FIGDIR, "orca12_vs_tripolar_grid_Nx$(NX)_Ny$(NY)_Nz$(NZ)_halo$(HALO[1]).png")
    save(outfile, fig, px_per_unit = 2)

    @info "Saved ORCA12 vs Tripolar grid comparison." outfile
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
