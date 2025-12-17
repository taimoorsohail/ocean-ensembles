module BasinMask

using ClimaOcean
using CUDA: @allowscalar
using Oceananigans
using Oceananigans.Fields: instantiate, location
using PolygonOps
using StaticArrays
using Oceananigans.Architectures: architecture

export basin_mask, get_coords_from_grid, section_mask

const SomeTripolarGrid = Union{TripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}}
const TripolarOrLatLonGrid = Union{SomeTripolarGrid, LatitudeLongitudeGrid}

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

    Nx = grid.Nx
    Ny = grid.Ny

    mask = zeros(Int, Nx, Ny)

    # All grid points in index space
    pts = collect(Iterators.product(1:Nx, 1:Ny))

    for (xs, ys, label) in zip(xs_list, ys_list, labels)
        @assert length(xs) == length(ys)

        # Build & close polygon
        poly = collect(zip(xs, ys))
        push!(poly, poly[1])

        inside = PolygonOps.inpolygon.(pts, Ref(poly))

        mask[reshape(inside .== 1, Nx, Ny)] .= label
    end

    return mask
end


end # module
