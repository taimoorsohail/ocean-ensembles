module BasinMask

    using NumericalEarth
    using CUDA: @allowscalar
    using Oceananigans
    using Oceananigans.Fields: instantiate, location
    using PolygonOps
    using StaticArrays
    using Oceananigans.Architectures: architecture

    export basin_mask, get_coords_from_grid, section_mask, regrid_bathymetry,
           apply_polygon_mask!

    const SomeTripolarGrid = Union{TripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}}
    const TripolarOrLatLonGrid = Union{SomeTripolarGrid, LatitudeLongitudeGrid}
    const DistributedGrid = Oceananigans.DistributedComputations.DistributedGrid

    function get_coords_from_grid(grid::SomeTripolarGrid, var)
        arch = architecture(grid)
        lons = λnodes(grid, instantiate.(location(var))..., with_halos=false)
        lats = φnodes(grid, instantiate.(location(var))..., with_halos=false)
        points = @allowscalar vec(SVector.(lons, lats))
        return lats, lons, points
    end

    function get_coords_from_grid(grid::LatitudeLongitudeGrid, var)
        arch = architecture(grid)
        lons = λnodes(grid, instantiate.(location(var))..., with_halos=false)
        lats = φnodes(grid, instantiate.(location(var))..., with_halos=false)

        X = [lons[i] for i in 1:grid.Nx, j in 1:grid.Ny]
        Y = [lats[j] for i in 1:grid.Nx, j in 1:grid.Ny]

        lons, lats = X, Y

        if arch == CPU()
            points = vec(SVector.(lons, lats))
        else
            points = @allowscalar vec(SVector.(lons, lats))
        end

        return lats, lons, points
    end

    function basin_mask(grid::TripolarOrLatLonGrid, basin::AbstractString, var::Oceananigans.Field)
        arch = architecture

        GlobalLonsPts=[0,360,360,0,0]
        GlobalLatsPts=[-90,-90,90,90,-90]

        IndLonsPts=[20, 20, 40,100, 100, 110,145,145,20];
        IndLatsPts=[-90, 28, 30, 30, 0, -10,-10,-90,-90];

        PacLonsPts=[145, 145, 110,100, 100, 260,260,300,300, 145];
        PacLatsPts=[-90, -10, -10, 0, 90, 90,20,0,-90,-90];

        # Atlantic is a bit more complicated
        AtleastLonsPts = [260,260,300,300, 360, 360, 260];
        AtleastLatsPts = [90,20,0,-90,-90, 90,90] ;
        AtlwestLonsPts = [0,20,20,0,0]
        AtlwestLatsPts = [-90,-90,28,30,-90] ;
        AtlarcticLonsPts = [0,20,20,0,0]
        AtlarcticLatsPts = [50,55,90,90,50]

        Globalpolygon = SVector.(GlobalLonsPts, GlobalLatsPts)
        Indpolygon = SVector.(IndLonsPts, IndLatsPts)    # boundary of the polygon
        Pacpolygon = SVector.(PacLonsPts, PacLatsPts)    # boundary of the polygon
        # Atlantic has multiple polygons due to lon grid bw 0 and 360
        Atleastpolygon = SVector.(AtleastLonsPts, AtleastLatsPts)    # boundary of the polygon
        Atlwestpolygon = SVector.(AtlwestLonsPts, AtlwestLatsPts)    # boundary of the polygon
        Atlarcticpolygon = SVector.(AtlarcticLonsPts, AtlarcticLatsPts)    # boundary of the polygon

        lats, lons, points = get_coords_from_grid(grid, var)

        if basin in ["indian", "Indian"]
            polygon = Indpolygon
            mask = @allowscalar (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

        elseif basin in ["pacific", "Pacific"]
            polygon = Pacpolygon
            mask = @allowscalar (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

        elseif basin in ["atlantic", "Atlantic"]
            polygon = Atleastpolygon
            mask_1 = @allowscalar (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

            polygon = Atlwestpolygon
            mask_2 = @allowscalar (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

            polygon = Atlarcticpolygon
            mask_3 = @allowscalar (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

            mask = mask_1 .+ mask_3 .+ mask_2

        elseif basin in ["indo-pacific", "Indo-pacific", "Indo-Pacific"]
            polygon = Indpolygon
            mask_1 = @allowscalar (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

            polygon = Pacpolygon
            mask_2 = @allowscalar (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

            mask = mask_1 .+ mask_2
        elseif isempty(basin)
            polygon = Globalpolygon
            mask = @allowscalar (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))
        else
            throw("Basin unknown, must be one of Indian, Atlantic, Pacific, or Indo-Pacific")
        end

        is_valid = maximum(mask) == 1
        is_valid || throw(ErrorException("Maximum value of mask is not 1"))
        bool_mask = convert(Array{Bool}, mask)

        return bool_mask
    end


    """
        section_mask(xs_list, ys_list, labels, grid)

    Create an integer label mask of size (grid.nx, grid.ny) from polygon vertex lists
    defined in index space.

    Arguments
    ---------
    - xs_list :: Vector{<:AbstractVector}
    - ys_list :: Vector{<:AbstractVector}
        Lists of x and y vertex coordinates (indices).
    - labels :: Vector{Int}
        Integer label assigned to each polygon.
    - grid
        Grid object with fields `nx` and `ny`.

    Returns
    -------
    - mask :: Matrix{Int}
        Integer mask with 0 = outside all polygons.
    """
    function section_mask(xs_list, ys_list, labels, grid)
        @assert length(xs_list) == length(ys_list) == length(labels)

        Nx, Ny = grid.Nx, grid.Ny
        mask = zeros(Int, Nx, Ny)

        # points in index space
        pts = [[i, j] for i in 1:Nx, j in 1:Ny]
        pts = reshape(pts, :)

        for (xs, ys, label) in zip([xs_list], [ys_list], labels)
            @assert length(xs) == length(ys)

            poly = [[xs[i], ys[i]] for i in eachindex(xs)]
            push!(poly, poly[1])   # close polygon

            inside = PolygonOps.inpolygon.(pts, Ref(poly))
            mask[reshape(inside .== 1, Nx, Ny)] .= label
        end

        return mask
    end


    function rank_y_bounds(rank, Ny, total_ranks)
        base, remainder = divrem(Ny, total_ranks)

        if rank < remainder
            local_ny = base + 1
            y_start = rank * (base + 1) + 1
        else
            local_ny = base
            y_start = remainder * (base + 1) + (rank - remainder) * base + 1
        end

        y_end = y_start + local_ny - 1
        return y_start, y_end
    end

    @inline point_inside_lower(y, ymin) = y ≥ ymin
    @inline point_inside_upper(y, ymax) = y ≤ ymax

    function clip_polygon_against_y(xs, ys, y_limit; keep_above)
        N = length(xs)
        N == 0 && return Float64[], Float64[]

        x_out = Float64[]
        y_out = Float64[]

        inside(y) = keep_above ? point_inside_lower(y, y_limit) : point_inside_upper(y, y_limit)

        for n in 1:N
            n_next = n == N ? 1 : n + 1

            x1 = Float64(xs[n])
            y1 = Float64(ys[n])
            x2 = Float64(xs[n_next])
            y2 = Float64(ys[n_next])

            inside1 = inside(y1)
            inside2 = inside(y2)

            if inside1 && inside2
                push!(x_out, x2)
                push!(y_out, y2)
            elseif inside1 && !inside2
                t = (y_limit - y1) / (y2 - y1)
                push!(x_out, x1 + t * (x2 - x1))
                push!(y_out, y_limit)
            elseif !inside1 && inside2
                t = (y_limit - y1) / (y2 - y1)
                push!(x_out, x1 + t * (x2 - x1))
                push!(y_out, y_limit)
                push!(x_out, x2)
                push!(y_out, y2)
            end
        end

        return x_out, y_out
    end

    function clip_polygon_to_rank(xs, ys, y_start, y_end)
        x_clip, y_clip = clip_polygon_against_y(xs, ys, y_start; keep_above=true)
        x_clip, y_clip = clip_polygon_against_y(x_clip, y_clip, y_end; keep_above=false)
        return x_clip, y_clip
    end

    function apply_polygon_mask!(bottom_height, underlying_grid, xs, ys, mask_id, replacement, condition, localrank, Ny, total_ranks, arch)
        y_start, y_end = rank_y_bounds(localrank, Ny, total_ranks)
        x_clip, y_clip = clip_polygon_to_rank(xs, ys, y_start, y_end)

        length(x_clip) < 3 && return nothing

        local_ny = y_end - y_start + 1
        x_local = clamp.(round.(Int, x_clip), 1, underlying_grid.Nx)
        y_local = clamp.(round.(Int, y_clip .- y_start .+ 1), 1, local_ny)

        mask_cpu = section_mask(x_local, y_local, fill(mask_id, length(x_local)), underlying_grid)

        bh = Array(interior(bottom_height)[:, :, 1])
        idx = (mask_cpu .== mask_id) .& condition(bh)
        bh[idx] .= replacement

        interior(bottom_height)[:, :, 1] .= bh

        return nothing
    end

    # function regrid_bathymetry_masked(target_grid::DistributedGrid, metadata;
    #                            height_above_water = nothing,
    #                            minimum_depth = 0,
    #                            interpolation_passes = 1,
    #                            major_basins = 1,
    #                            mask = nothing,
    #                            height_threshold = nothing)

    #     download_dataset(metadata)

    #     global_grid = reconstruct_global_grid(target_grid)
    #     global_grid = on_architecture(CPU(), global_grid)
    #     arch = architecture(target_grid)
    #     Nx, Ny, _ = size(global_grid)

    #     # If all ranks open a gigantic bathymetry and the memory is
    #     # shared, we could easily have OOM errors.
    #     # We perform the reconstruction only on rank 0 and share the result.
    #     bottom_height = if arch.local_rank == 0
    #         # use regrid method that assumes data is downloaded
    #         bottom_field = _regrid_bathymetry(global_grid, metadata;
    #                                           height_above_water, minimum_depth, interpolation_passes, major_basins)
    #         bottom_field.data[1:Nx, 1:Ny, 1]
    #         if mask !== nothing
    #             mask_bathymetry!
    #     else
    #         zeros(Nx, Ny)
    #     end

    #     # Synchronize
    #     Oceananigans.DistributedComputations.barrier(arch.communicator)

    #     # Share the result (can we share SubArrays?)
    #     bottom_height = all_reduce(+, bottom_height, arch)

    #     # Partition the result
    #     local_bottom_height = Field{Center, Center, Nothing}(target_grid)
    #     set!(local_bottom_height, bottom_height)
    #     fill_halo_regions!(local_bottom_height)

    #     return local_bottom_height
    # end

end # module
