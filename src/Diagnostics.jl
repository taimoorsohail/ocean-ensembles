module Diagnostics

    using OceanEnsembles
    using Oceananigans
    using Oceananigans.Fields: location, ReducedField
    using Oceananigans.Grids: AbstractGrid

    using PyCall
    using SparseArrays
    using LinearAlgebra

    export ocean_tracer_content!, volume_transport!, regrid_tracers!, regridder_weights

    """
        regridder_weights!(source_field, destination_field; method = "conservative")
    
    Regrid the `source_field` onto the `destination_field` using the specified method.
    xESMF exposes five different regridding algorithms from the ESMF library, specified with the `method` keyword argument:

    bilinear: ESMF.RegridMethod.BILINEAR
    conservative: ESMF.RegridMethod.CONSERVE
    conservative_normed: ESMF.RegridMethod.CONSERVE
    patch: ESMF.RegridMethod.PATCH
    nearest_s2d: ESMF.RegridMethod.NEAREST_STOD
    nearest_d2s: ESMF.RegridMethod.NEAREST_DTOS

    where conservative_normed is just the conservative method with the normalization set to ESMF.NormType.FRACAREA instead of the default norm_type=ESMF.NormType.DSTAREA.
    For more information, see the xESMF documentation: https://xesmf.readthedocs.io/en/latest/notebooks/Compare_algorithms.html
    """

    const SomeTripolarGrid = Union{TripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}}
    const SomeLatitudeLongitudeGrid = Union{LatitudeLongitudeGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid}}
    const TripolarOrLatLonGrid = Union{SomeTripolarGrid, SomeLatitudeLongitudeGrid}

    function two_dimensionalize(lat::AbstractVector, lon::AbstractVector) 
        Nx = length(lon)
        Ny = length(lat)
        lat = repeat(lat', Nx)
        lon = repeat(lon, 1, Ny)

        return lat, lon
    end

    function coordinate_dataset(grid::SomeLatitudeLongitudeGrid)
        lat = Array(φnodes(grid, Center(), Center(), Center()))
        lon = Array(λnodes(grid, Center(), Center(), Center()))

        lat_b = Array(φnodes(grid, Face(), Face(), Center()))
        lon_b = Array(grid.λᶠᵃᵃ[1:grid.Nx+1])

        lat,   lon   = two_dimensionalize(lat,   lon)
        lat_b, lon_b = two_dimensionalize(lat_b, lon_b)

        return structured_coordinate_dataset(lat, lon, lat_b, lon_b)
    end

    function coordinate_dataset(grid::SomeTripolarGrid)
        lat = Array(grid.φᶜᶜᵃ[1:grid.Nx, 1:grid.Ny])
        lon = Array(grid.λᶜᶜᵃ[1:grid.Nx, 1:grid.Ny])

        lat_b = Array(grid.φᶠᶠᵃ[1:grid.Nx+1, 1:grid.Ny+1])
        lon_b = Array(grid.λᶠᶠᵃ[1:grid.Nx+1, 1:grid.Ny+1])

        return structured_coordinate_dataset(lat, lon, lat_b, lon_b)
    end

    function structured_coordinate_dataset(lat, lon, lat_b, lon_b)
        numpy  = OceanEnsembles.get_np()
        xarray = OceanEnsembles.get_xr()

        lat = numpy.array(lat)
        lon = numpy.array(lon)

        lat_b = numpy.array(lat_b)
        lon_b = numpy.array(lon_b)

        ds_lat = xarray.DataArray(
            lat',
            dims=["y", "x"],
            coords= PyObject(Dict(
                "lat" => (["y", "x"], lat'),
                "lon" => (["y", "x"], lon')
            )),
            name="latitude"
        )

        ds_lon = xarray.DataArray(
            lon',
            dims=["y", "x"],
            coords= PyObject(Dict(
                "lat" => (["y", "x"], lat'),
                "lon" => (["y", "x"], lon')
            )),
            name="longitude"
        )

        ds_lat_b = xarray.DataArray(
            lat_b',
            dims=["y_b", "x_b"],
            coords= PyObject(Dict(
                "lat_b" => (["y_b", "x_b"], lat_b'),
                "lon_b" => (["y_b", "x_b"], lon_b')
            )),
        )

        ds_lon_b = xarray.DataArray(
            lon_b',
            dims=["y_b", "x_b"],
            coords= PyObject(Dict(
                "lat_b" => (["y_b", "x_b"], lat_b'),
                "lon_b" => (["y_b", "x_b"], lon_b')
            )),
        )

        return  xarray.Dataset(
            PyObject(
                Dict("lat"   => ds_lat, 
                    "lon"   => ds_lon,
                    "lat_b" => ds_lat_b,
                    "lon_b" => ds_lon_b))
        )
    end


    function regridder_weights(
    source_field::Field, 
    destination_field::Field; 
    method::String = "conservative")

        # Create source and destination fields

        src_ds = coordinate_dataset(source_field.grid)
        dst_ds = coordinate_dataset(destination_field.grid)

        # # Extract Centers
        # if isa(source_field.grid, SomeTripolarGrid)
        #     src_lats = src.grid.φᶜᶜᵃ[1:src.grid.Nx, 1:src.grid.Ny]
        #     src_lons = src.grid.λᶜᶜᵃ[1:src.grid.Nx, 1:src.grid.Ny]
        # elseif isa(source_field.grid, SomeLatitudeLongitudeGrid)
        #     src_lats = src.grid.φᵃᶜᵃ[1:src.grid.Ny]
        #     src_lons = src.grid.λᶜᵃᵃ[1:src.grid.Nx]
        # else
        #     error("Unsupported grid type for source grid")
        # end

        # if isa(destination_field.grid, SomeTripolarGrid)
        #     dst_lats = dst.grid.φᶜᶜᵃ[1:dst.grid.Nx, 1:dst.grid.Ny]
        #     dst_lons = dst.grid.λᶜᶜᵃ[1:dst.grid.Nx, 1:dst.grid.Ny]
        # elseif isa(destination_field.grid, SomeLatitudeLongitudeGrid)
        #     dst_lats = dst.grid.φᵃᶜᵃ[1:dst.grid.Ny]
        #     dst_lons = dst.grid.λᶜᵃᵃ[1:dst.grid.Nx]
        # else
        #     error("Unsupported grid type for destination grid")
        # end

        # # Extract corners
        # if isa(source_field.grid, SomeTripolarGrid)
        #     src_lats_b = src.grid.φᶠᶠᵃ[1:src.grid.Nx+1, 1:src.grid.Ny+1]
        #     src_lons_b = src.grid.λᶠᶠᵃ[1:src.grid.Nx+1, 1:src.grid.Ny+1]
        # elseif isa(source_field.grid, SomeLatitudeLongitudeGrid)
        #     src_lats_b = src.grid.φᵃᶠᵃ[1:src.grid.Ny+1]
        #     src_lons_b = src.grid.λᶠᵃᵃ[1:src.grid.Nx+1]
        # else
        #     error("Unsupported grid type for source grid")
        # end

        # if isa(destination_field.grid, SomeTripolarGrid)
        #     dst_lats_b = dst.grid.φᶠᶠᵃ[1:dst.grid.Nx+1, 1:dst.grid.Ny+1]
        #     dst_lons_b = dst.grid.λᶠᶠᵃ[1:dst.grid.Nx+1, 1:dst.grid.Ny+1]
        # elseif isa(destination_field.grid, SomeLatitudeLongitudeGrid)
        #     dst_lats_b = dst.grid.φᵃᶠᵃ[1:dst.grid.Ny+1]
        #     dst_lons_b = dst.grid.λᶠᵃᵃ[1:dst.grid.Nx+1]
        # else
        #     error("Unsupported grid type for destination grid")
        # end

        # # Move to Python
        # # Convert to numpy arrays for xesmf
        # lat_src_np = OceanEnsembles.get_np().array(src_lats)
        # lon_src_np = OceanEnsembles.get_np().array(src_lons)

        # lat_dst_np = OceanEnsembles.get_np().array(dst_lats)
        # lon_dst_np = OceanEnsembles.get_np().array(dst_lons)

        # lat_src_b_np = OceanEnsembles.get_np().array(src_lats_b)
        # lon_src_b_np = OceanEnsembles.get_np().array(src_lons_b)

        # lat_dst_b_np = OceanEnsembles.get_np().array(dst_lats_b)
        # lon_dst_b_np = OceanEnsembles.get_np().array(dst_lons_b)

        # # Create xarray DataArrays for source and destination grids
        # dst_data = Field{Center, Center, Nothing}(destination_field.grid)
        # src_data = Field{Center, Center, Nothing}(source_field.grid)

        # src_np = OceanEnsembles.get_np().squeeze(OceanEnsembles.get_np().array(collect(interior(src_data))))
        # dst_np = OceanEnsembles.get_np().squeeze(OceanEnsembles.get_np().array(collect(interior(dst_data))))

        # src_da = OceanEnsembles.get_xr().DataArray(
        #     src_np,
        #     dims=["y", "x"],
        #     coords= PyObject(Dict(
        #         "lat" => (["y", "x"], lat_src_np),
        #         "lon" => (["y", "x"], lon_src_np)
        #     )),
        #     name="source"
        # )

        # src_b_da = OceanEnsembles.get_xr().DataArray(
        #     lat_src_b_np,
        #     dims=["y_b", "x_b"],
        #     coords= PyObject(Dict(
        #         "lat_b" => (["y_b", "x_b"], lat_src_b_np),
        #         "lon_b" => (["y_b", "x_b"], lon_src_b_np)
        #     )),
        #     name="bounds"
        # )

        # src_ds = OceanEnsembles.get_xr().Dataset(
        #     PyObject(Dict("source" => src_da, "bounds" => src_b_da))
        # )

        # dst_ds = OceanEnsembles.get_xr().DataArray(
        #     dst_np,
        #     dims=["lon", "lat"],
        #     coords= PyObject(Dict(
        #         "lat" => (["lat"], lat_dst_np),
        #         "lon" => (["lon"], lon_dst_np)    )),
        #     name="destination"
        # )

        regridder = OceanEnsembles.get_xesmf().Regridder(src_ds, dst_ds, method)#, periodic=PyObject(true))

        # Move back to Julia
        # Convert the regridder weights to a Julia sparse matrix
        coo = regridder.weights.data
        coords = coo[:coords]
        rows = coords[1,:].+1
        cols = coords[2,:].+1
        vals = Float64.(coo[:data])

        shape = Tuple(Int.(coo[:shape]))
        W = sparse(rows, cols, vals, shape[1], shape[2])

        return W
    end

    """
        regrid_tracers!(src::Field, dst::Field, W::SparseMatrixCSC)
    Regrid the `src` field onto the `dst` field using the provided weights `W`.
    The function assumes that the vertical grid (z) of both fields is the same.
    """
    function regrid_tracers!(dst::Field, W::SparseMatrixCSC, src::Field)

        @assert dst.grid.z.cᵃᵃᶜ[1:dst.grid.Nz] == src.grid.z.cᵃᵃᶜ[1:src.grid.Nz] "Source and destination grids must have exactly the same vertical grid (z)."
        
        # Perform regridding
        for k in 1:dst.grid.Nz
            # Flatten the source field for regridding
            src_flat = vec(collect(interior(src))[:,:,k])  # shape (ncell, 1)

            # Regrid the source field to the destination grid
            # LinearAlgebra.mul!(vec(interior(dst, :, :, k)), W, vec(interior(src, :, :, k)))            
            dst_vec = reshape(W * src_flat, dst.grid.Nx, dst.grid.Ny)

            # # Fill the destination field with the regridded values
            interior(dst)[:,:,k] .= dst_vec
        end
        return nothing
    end

    """
        ocean_tracer_content!(names, ∫outputs; dst_field::Field, weights::SparseMatrixCSC, outputs, operator, dims, condition, suffix::AbstractString)
    Compute the integral of the tracers in `outputs` using the specified `operator` and over the specified `dims`.
    The `condition` can be used to specify a mask for the integral.
    The `suffix` is appended to the names of the output fields.
    """

    function ocean_tracer_content!(names, ∫outputs; dst_field::Field = nothing, weights::SparseMatrixCSC = nothing, outputs, operator = nothing, dims, condition = nothing, suffix::AbstractString = "")
        if dst_field != nothing
            onefield = CenterField(dst_field.grid)
        else
            onefield = CenterField(f.grid)
        end

        set!(onefield, 1)

        if operator == nothing
            operator = onefield
        end

        for key in keys(outputs)
            f = outputs[key]
            if dst_field != nothing
                f_dst = regrid_tracers!(f, dst_field, weights)
                ∫f = Integral(f_dst * operator; dims, condition)
            else
                ∫f = Integral(f * operator; dims, condition)
            end
            push!(∫outputs, ∫f)
            push!(names, Symbol(key, suffix))
        end

        ∫dV = Integral(onefield * operator; dims, condition)
        push!(∫outputs, ∫dV)
        push!(names, Symbol(:dV, suffix))

        return names, ∫outputs
    end

    """
        volume_transport!(names, ∫outputs; outputs, operators, dims, condition, suffix::AbstractString)
    Compute the volume transport of the velocities in `outputs` using the specified `operators` and over the specified `dims`.
    The `condition` can be used to specify a mask for the transport.
    The `suffix` is appended to the names of the output fields.
    """ 

    function volume_transport!(names, ∫outputs; outputs, operators, dims, condition, suffix::AbstractString)
        if length(outputs) == length(operators)
            for (i, key) in enumerate(keys(outputs))
                if key == :w
                    cond3d = condition[2]
                else
                    cond3d = condition[1]
                end

                f = outputs[key]
                
                ∫f = sum(f * operators[i]; dims, condition = cond3d)
                push!(∫outputs, ∫f)
                push!(names, Symbol(key, suffix))

                LX, LY, LZ = location(f)
                onefield = Field{LX, LY, LZ}(f.grid)
                set!(onefield, 1)
                
                ∫dV = sum(onefield * operators[i]; dims, condition = cond3d)
                push!(∫outputs, ∫dV)
                push!(names, Symbol(:dV, "_$(key)", suffix))
            end
        else
            throw(ArgumentError("The number of operators must be equal to the number of tracers"))
        end
        return names, ∫outputs
    end
end
