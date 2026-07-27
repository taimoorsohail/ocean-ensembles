using JLD2
using Glob
using Logging
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: Center, Face, λnodes, φnodes, znodes
using WriteVTK

const SAVED_PATH = joinpath(@__DIR__, "..", "outputs", "saved")
const INPUT_PATTERN = "combined_global_75_fields_sxtdeg_RYF_run*.jld2"
const SURFACE_PATTERN = "combined_global_surface_fluxes_sxtdeg_RYF_run*.jld2"
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "..", "outputs", "saved", "vtk")
const EARTH_RADIUS = 6_371_000f0
const MODEL_LEVEL = 75
const SNAPSHOTS_PER_YEAR = 365

function print_usage()
    println("Usage: julia --project=ocean-ensembles/experiments/submission_scripts ocean-ensembles/analysis/jld2_to_vtk.jl [OUTPUT_PREFIX]")
    println()
    println("Examples:")
    println("  julia --project=ocean-ensembles/experiments/submission_scripts ocean-ensembles/analysis/jld2_to_vtk.jl")
    println("  julia --project=ocean-ensembles/experiments/submission_scripts ocean-ensembles/analysis/jld2_to_vtk.jl /tmp/global_75_last_year")
    println()
    println("Exports the final 365 snapshots and writes a .pvd ParaView collection plus 365 .vts files.")
    return nothing
end

run_number(file::AbstractString) = begin
    match_result = match(r"_run(\d+)\.jld2$", basename(file))
    isnothing(match_result) ? -1 : parse(Int, match_result.captures[1])
end

function run_files(pattern::AbstractString)
    files = glob(pattern, SAVED_PATH)
    filter!(file -> run_number(file) >= 0, files)
    sort!(files; by = run_number)
    isempty(files) && error("No files matching $pattern were found in $SAVED_PATH.")
    return files
end

function sorted_frame_keys(file)
    haskey(file, "timeseries/t") || error("Input file is missing timeseries/t.")
    frame_keys = sort!(parse.(Int, String.(collect(keys(file["timeseries/t"])))))
    isempty(frame_keys) && error("Input file contains no time frames.")
    return frame_keys
end

function time_records(pattern::AbstractString)
    records_by_time = Dict{Float64, NamedTuple{(:path, :key, :run), Tuple{String, Int, Int}}}()

    for path in run_files(pattern)
        run = run_number(path)
        jldopen(path, "r") do file
            for key in sorted_frame_keys(file)
                time = Float64(file["timeseries/t/$key"])
                records_by_time[time] = (; path, key, run)
            end
        end
    end

    times = sort!(collect(keys(records_by_time)))
    return [(; time, records_by_time[time]...) for time in times]
end

function read_2d_field(file, variable::AbstractString, frame_key::Int)
    path = "timeseries/$variable/$frame_key"
    haskey(file, path) || error("Input file is missing $path.")
    data = file[path]

    if ndims(data) == 2
        return Float32.(data)
    elseif ndims(data) == 3 && size(data, 3) == 1
        return Float32.(dropdims(data; dims = 3))
    end

    error("Expected $path to contain one horizontal plane, got size $(size(data)).")
end

function nearest_record(records, times::Vector{Float64}, target_time::Float64)
    upper = clamp(searchsortedfirst(times, target_time), 1, length(times))
    lower = max(1, upper - 1)
    index = abs(times[lower] - target_time) <= abs(times[upper] - target_time) ? lower : upper
    return records[index]
end

const OFFSET_ARRAYS = Base.loaded_modules[Base.PkgId(Base.UUID("6fe1bfb0-de20-5000-8ca7-80f57d26f881"), "OffsetArrays")]

underlying_grid(grid) = hasproperty(grid, :underlying_grid) ? getproperty(grid, :underlying_grid) : grid

function materialize_offset_array(source)
    if hasproperty(source, :parent) && hasproperty(source, :offsets)
        return OFFSET_ARRAYS.OffsetArray(getproperty(source, :parent), getproperty(source, :offsets)...)
    end
    return source
end

function materialized_underlying_grid(grid)
    g = underlying_grid(grid)

    return OrthogonalSphericalShellGrid{Periodic, RightFaceFolded, Bounded}(CPU(),
        getproperty(g, :Nx), getproperty(g, :Ny), getproperty(g, :Nz),
        getproperty(g, :Hx), getproperty(g, :Hy), getproperty(g, :Hz),
        getproperty(g, :Lz),
        materialize_offset_array(getproperty(g, :λᶜᶜᵃ)),
        materialize_offset_array(getproperty(g, :λᶠᶜᵃ)),
        materialize_offset_array(getproperty(g, :λᶜᶠᵃ)),
        materialize_offset_array(getproperty(g, :λᶠᶠᵃ)),
        materialize_offset_array(getproperty(g, :φᶜᶜᵃ)),
        materialize_offset_array(getproperty(g, :φᶠᶜᵃ)),
        materialize_offset_array(getproperty(g, :φᶜᶠᵃ)),
        materialize_offset_array(getproperty(g, :φᶠᶠᵃ)),
        getproperty(g, :z),
        materialize_offset_array(getproperty(g, :Δxᶜᶜᵃ)),
        materialize_offset_array(getproperty(g, :Δxᶠᶜᵃ)),
        materialize_offset_array(getproperty(g, :Δxᶜᶠᵃ)),
        materialize_offset_array(getproperty(g, :Δxᶠᶠᵃ)),
        materialize_offset_array(getproperty(g, :Δyᶜᶜᵃ)),
        materialize_offset_array(getproperty(g, :Δyᶠᶜᵃ)),
        materialize_offset_array(getproperty(g, :Δyᶜᶠᵃ)),
        materialize_offset_array(getproperty(g, :Δyᶠᶠᵃ)),
        materialize_offset_array(getproperty(g, :Azᶜᶜᵃ)),
        materialize_offset_array(getproperty(g, :Azᶠᶜᵃ)),
        materialize_offset_array(getproperty(g, :Azᶜᶠᵃ)),
        materialize_offset_array(getproperty(g, :Azᶠᶠᵃ)),
        getproperty(g, :radius),
        getproperty(g, :conformal_mapping))
end

@inline surface_matrix_3d(A::AbstractMatrix) = reshape(A, size(A, 1), size(A, 2), 1)

function speed_workspace(grid)
    speed_grid = materialized_underlying_grid(grid)
    ufield = XFaceField(speed_grid)
    vfield = YFaceField(speed_grid)
    speed_field = @at (Center, Center, Nothing) sqrt(ufield^2 + vfield^2) |> Field
    return (; ufield, vfield, speed_field)
end

function horizontal_speed(u::AbstractMatrix, v::AbstractMatrix, workspace)
    set!(workspace.ufield, surface_matrix_3d(u))
    set!(workspace.vfield, surface_matrix_3d(v))
    fill_halo_regions!(workspace.ufield)
    fill_halo_regions!(workspace.vfield)
    compute!(workspace.speed_field)
    return Float32.(Array(interior(workspace.speed_field)[:, :, 1]))
end

function spherical_coordinates(grid, level::Int)
    underlying = hasproperty(grid, :underlying_grid) ? grid.underlying_grid : grid
    1 <= level <= underlying.Nz || error("Model level $level is outside 1:$(underlying.Nz).")

    longitude = Float32.(λnodes(underlying, Center(), Center(), Center(); with_halos = false))
    latitude = Float32.(φnodes(underlying, Center(), Center(), Center(); with_halos = false))
    face_longitude = Float32.(λnodes(underlying, Center(), Face(), Center(); with_halos = false))
    face_latitude = Float32.(φnodes(underlying, Center(), Face(), Center(); with_halos = false))
    north_longitude = face_longitude[:, end:end]
    north_latitude = face_latitude[:, end:end]
    z_levels = znodes(underlying, Center(), Center(), Center(); with_halos = false)
    return longitude, latitude, north_longitude, north_latitude, Float32(z_levels[level])
end

function close_visualization_field(field::AbstractMatrix)
    # Duplicate the northernmost values onto the folded face, repeat the
    # periodic edge, and fill the synthetic Antarctic cap. The duplicated
    # values are for visualization only.
    field_with_fold = hcat(field, view(field, :, size(field, 2):size(field, 2)))
    periodic_field = vcat(field_with_fold, view(field_with_fold, 1:1, :))
    return hcat(view(periodic_field, :, 1:1), periodic_field)
end

function close_land_mask(land_mask::BitMatrix)
    mask_with_fold = hcat(land_mask, view(land_mask, :, size(land_mask, 2):size(land_mask, 2)))
    periodic_mask = vcat(mask_with_fold, view(mask_with_fold, 1:1, :))
    antarctic_cap = trues(size(periodic_mask, 1), 1)
    return BitMatrix(hcat(antarctic_cap, periodic_mask))
end

function nan_land(field::AbstractMatrix, closed_land_mask::BitMatrix)
    closed_field = close_visualization_field(field)
    closed_field[closed_land_mask] .= NaN32
    return closed_field
end

parent_array(source) = hasproperty(source, :parent) ? getproperty(source, :parent) : Array(source)

function interior_start(source, dim::Int, fallback_halo::Int, interior_size::Int, stored_size::Int)
    if hasproperty(source, :offsets)
        offsets = getproperty(source, :offsets)
        if dim <= length(offsets)
            start = 1 - offsets[dim]
            1 <= start <= stored_size - interior_size + 1 && return start
        end
    end

    stored_size == interior_size && return 1
    start = fallback_halo + 1
    1 <= start <= stored_size - interior_size + 1 && return start
    error("Could not crop stored dimension $stored_size to interior size $interior_size.")
end

function surface_land_mask(grid)
    hasproperty(grid, :immersed_boundary) || error("Grid does not contain an immersed boundary.")
    underlying = underlying_grid(grid)
    bottom_height = getproperty(grid.immersed_boundary, :bottom_height)
    source = hasproperty(bottom_height, :data) ? getproperty(bottom_height, :data) : bottom_height
    data = parent_array(source)
    i0 = interior_start(source, 1, underlying.Hx, underlying.Nx, size(data, 1))
    j0 = interior_start(source, 2, underlying.Hy, underlying.Ny, size(data, 2))
    bottom = ndims(data) == 2 ?
             view(data, i0:i0+underlying.Nx-1, j0:j0+underlying.Ny-1) :
             view(data, i0:i0+underlying.Nx-1, j0:j0+underlying.Ny-1, 1)
    return BitMatrix(.!(bottom .< 0f0))
end

function close_visualization_coordinates(longitude::AbstractMatrix,
                                         latitude::AbstractMatrix,
                                         north_longitude::AbstractMatrix,
                                         north_latitude::AbstractMatrix,
                                         depth::Float32)
    longitude_with_fold = hcat(longitude, north_longitude)
    latitude_with_fold = hcat(latitude, north_latitude)
    periodic_longitude = vcat(longitude_with_fold, view(longitude_with_fold, 1:1, :))
    periodic_latitude = vcat(latitude_with_fold, view(latitude_with_fold, 1:1, :))
    closed_longitude = hcat(view(periodic_longitude, :, 1:1), periodic_longitude)
    south_pole = fill(-90f0, size(periodic_latitude, 1), 1)
    closed_latitude = hcat(south_pole, periodic_latitude)

    radius = EARTH_RADIUS + depth
    λ = deg2rad.(closed_longitude)
    φ = deg2rad.(closed_latitude)
    cosφ = cos.(φ)
    x = radius .* cosφ .* cos.(λ)
    y = radius .* cosφ .* sin.(λ)
    z = radius .* sin.(φ)

    return x, y, z, closed_longitude, closed_latitude
end

function convert_last_year(output_prefix::Union{Nothing, AbstractString} = nothing)
    depth_records = time_records(INPUT_PATTERN)
    length(depth_records) >= SNAPSHOTS_PER_YEAR ||
        error("Found only $(length(depth_records)) snapshots; $SNAPSHOTS_PER_YEAR are required.")
    depth_records = depth_records[end-SNAPSHOTS_PER_YEAR+1:end]

    surface_records = time_records(SURFACE_PATTERN)
    surface_times = Float64[record.time for record in surface_records]

    grid = with_logger(NullLogger()) do
        jldopen(last(depth_records).path, "r") do file
            haskey(file, "serialized/grid") || error("Input file is missing serialized/grid.")
            file["serialized/grid"]
        end
    end

    underlying = underlying_grid(grid)
    Nx, Ny = underlying.Nx, underlying.Ny
    longitude, latitude, north_longitude, north_latitude, depth = spherical_coordinates(grid, MODEL_LEVEL)
    size(longitude) == (Nx, Ny) || error("Grid coordinates have size $(size(longitude)); expected $((Nx, Ny)).")
    size(north_longitude) == (Nx, 1) || error("Northern fold coordinates have size $(size(north_longitude)); expected $((Nx, 1)).")

    land_mask = surface_land_mask(grid)
    size(land_mask) == (Nx, Ny) || error("Land mask has size $(size(land_mask)); expected $((Nx, Ny)).")
    vtk_land_mask = close_land_mask(land_mask)
    workspace = speed_workspace(grid)

    x, y, z, vtk_longitude, vtk_latitude = close_visualization_coordinates(longitude,
                                                                            latitude,
                                                                            north_longitude,
                                                                            north_latitude,
                                                                            depth)
    vtk_size = size(x)
    x3 = reshape(x, vtk_size..., 1)
    y3 = reshape(y, vtk_size..., 1)
    z3 = reshape(z, vtk_size..., 1)

    if isnothing(output_prefix)
        output_prefix = joinpath(DEFAULT_OUTPUT_DIR, "global_75_last_year")
    else
        output_prefix = abspath(expanduser(output_prefix))
        output_prefix = replace(output_prefix, r"\.pvd$" => "")
    end
    mkpath(dirname(output_prefix))

    duration_days = (last(depth_records).time - first(depth_records).time) / 86400
    gap_days = diff(Float64[record.time for record in depth_records]) ./ 86400
    large_gaps = gap_days[gap_days .> 1.5]
    !isempty(large_gaps) && @warn "The final 365 available snapshots contain missing daily outputs." gap_days = large_gaps
    @info "Exporting final 365 available snapshots" snapshots = length(depth_records) duration_days output_prefix

    output_files = paraview_collection(output_prefix) do collection
        depth_file = nothing
        depth_path = ""
        surface_file = nothing
        surface_path = ""
        progress_stride = max(1, cld(length(depth_records), 20))

        try
            for (index, record) in enumerate(depth_records)
                if record.path != depth_path
                    !isnothing(depth_file) && close(depth_file)
                    depth_file = jldopen(record.path, "r")
                    depth_path = record.path
                end

                fields = (
                    T = read_2d_field(depth_file, "T", record.key),
                    S = read_2d_field(depth_file, "S", record.key),
                    e = read_2d_field(depth_file, "e", record.key),
                    u = read_2d_field(depth_file, "u", record.key),
                    v = read_2d_field(depth_file, "v", record.key),
                    w = read_2d_field(depth_file, "w", record.key),
                )

                for variable in (:T, :S, :e, :u, :w)
                    size(getproperty(fields, variable)) == (Nx, Ny) ||
                        error("$variable has size $(size(getproperty(fields, variable))); expected $((Nx, Ny)).")
                end
                size(fields.v) == (Nx, Ny + 1) || error("v has size $(size(fields.v)); expected $((Nx, Ny + 1)).")

                surface_record = nearest_record(surface_records, surface_times, record.time)
                if surface_record.path != surface_path
                    !isnothing(surface_file) && close(surface_file)
                    surface_file = jldopen(surface_record.path, "r")
                    surface_path = surface_record.path
                end

                ice_concentration = read_2d_field(surface_file, "ice_concentration", surface_record.key)
                sea_surface_height = read_2d_field(surface_file, "surface_height", surface_record.key)
                size(ice_concentration) == (Nx, Ny) || error("Sea-ice concentration has size $(size(ice_concentration)); expected $((Nx, Ny)).")
                size(sea_surface_height) == (Nx, Ny) || error("Sea-surface height has size $(size(sea_surface_height)); expected $((Nx, Ny)).")

                speed = horizontal_speed(fields.u, fields.v, workspace)
                size(speed) == (Nx, Ny) || error("Centered speed has size $(size(speed)); expected $((Nx, Ny)).")

                frame_prefix = output_prefix * "_frame" * lpad(index, 3, '0')
                vtk_grid(frame_prefix, x3, y3, z3) do vtk
                    vtk["temperature", VTKPointData()] = nan_land(fields.T, vtk_land_mask)
                    vtk["salinity", VTKPointData()] = nan_land(fields.S, vtk_land_mask)
                    vtk["turbulent_kinetic_energy", VTKPointData()] = nan_land(fields.e, vtk_land_mask)
                    vtk["speed", VTKPointData()] = nan_land(speed, vtk_land_mask)
                    vtk["w", VTKPointData()] = nan_land(fields.w, vtk_land_mask)
                    vtk["sea_ice_concentration", VTKPointData()] = nan_land(ice_concentration, vtk_land_mask)
                    vtk["sea_surface_height", VTKPointData()] = nan_land(sea_surface_height, vtk_land_mask)
                    vtk["land_mask", VTKPointData()] = Float32.(vtk_land_mask)
                    vtk["longitude", VTKPointData()] = vtk_longitude
                    vtk["latitude", VTKPointData()] = vtk_latitude
                    vtk["depth", VTKPointData()] = fill(depth, vtk_size)
                    vtk["time_seconds", VTKFieldData()] = [record.time]
                    vtk["frame_key", VTKFieldData()] = [record.key]
                    vtk["source_run", VTKFieldData()] = [record.run]
                    vtk["surface_time_seconds", VTKFieldData()] = [surface_record.time]
                    vtk["surface_frame_key", VTKFieldData()] = [surface_record.key]
                    vtk["surface_source_run", VTKFieldData()] = [surface_record.run]
                    collection[record.time] = vtk
                end

                if index == 1 || index == length(depth_records) || index % progress_stride == 0
                    @info "VTK export progress" snapshot = index total = length(depth_records) model_time = record.time
                end
            end
        finally
            !isnothing(depth_file) && close(depth_file)
            !isnothing(surface_file) && close(surface_file)
        end
    end

    println("Wrote $(length(depth_records)) snapshots and ParaView collection:")
    println("  ", first(output_files))
    return output_files
end

convert_frame(output_prefix::Union{Nothing, AbstractString} = nothing) = convert_last_year(output_prefix)

function main(args::Vector{String})
    any(arg -> arg in ("-h", "--help"), args) && return print_usage()
    length(args) <= 1 || throw(ArgumentError("Expected at most one OUTPUT_PREFIX."))
    output_prefix = isempty(args) ? nothing : args[1]
    convert_last_year(output_prefix)
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        main(ARGS)
    catch err
        if err isa InterruptException
            rethrow()
        end
        showerror(stderr, err)
        println(stderr)
        print_usage()
        exit(1)
    end
end
