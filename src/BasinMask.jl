module BasinMask

using ClimaOcean
using CUDA
using Oceananigans
using Oceananigans.Fields: instantiate, location
using PolygonOps
using StaticArrays
using Oceananigans.Architectures: architecture

export basin_mask, get_coords_from_grid

const SomeTripolarGrid = Union{TripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}}
const TripolarOrLatLonGrid = Union{SomeTripolarGrid, LatitudeLongitudeGrid}

function get_coords_from_grid(grid::SomeTripolarGrid, var)
    arch = architecture(grid)
    lons = λnodes(grid, instantiate.(location(var))..., with_halos=false)
    lats = φnodes(grid, instantiate.(location(var))..., with_halos=false)
    points = CUDA.@allowscalar vec(SVector.(lons, lats))
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
        points = CUDA.@allowscalar vec(SVector.(lons, lats))
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
        mask = (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

    elseif basin in ["pacific", "Pacific"]
        polygon = Pacpolygon
        mask = (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

    elseif basin in ["atlantic", "Atlantic"]
        polygon = Atleastpolygon
        mask_1 = (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

        polygon = Atlwestpolygon
        mask_2 = (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

        polygon = Atlarcticpolygon
        mask_3 = (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

        mask = mask_1 .+ mask_3 .+ mask_2

    elseif basin in ["indo-pacific", "Indo-pacific", "Indo-Pacific"]
        polygon = Indpolygon
        mask_1 = (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

        polygon = Pacpolygon
        mask_2 = (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))

        mask = mask_1 .+ mask_2
    elseif isempty(basin)
        polygon = Globalpolygon
        mask = (reshape([inpolygon(p, polygon; in=true, on=false, out=false) for p in points], size(lats)))
    else
        throw("Basin unknown, must be one of Indian, Atlantic, Pacific, or Indo-Pacific")
    end

    is_valid = maximum(mask) == 1
    is_valid || throw(ErrorException("Maximum value of mask is not 1"))
    bool_mask = convert(Array{Bool}, mask)

    return bool_mask
end

end # module
