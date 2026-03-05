using NumericalEarth
using Oceananigans
using Oceananigans.Units
using OceanEnsembles
using CairoMakie
using Glob 
using Oceananigans.Fields: location
using JLD2

output_path = expanduser("../../outputs/saved_fields/")
figdir = expanduser("../../figures/")

resolution = "sxtdeg"

grid = create_grid(output_path * "global_75_fields_$(resolution)_RYF_iteration0", [0,1,2,3]; gridtype="TripolarGrid")

files_combined = filter(f -> !occursin("_rank", f),
                        glob("global_*$(resolution)_RYF_iteration*.jld2", output_path))

Tfield = CenterField(grid)
ufield = XFaceField(grid)
vfield = YFaceField(grid)

function spherical_coordinates_viz(λ, φ, r=1)
    # Convert degrees to radians
    λ_rad = deg2rad.(λ)
    φ_rad = deg2rad.(φ)

    x = @. r * cos(φ_rad) * cos(λ_rad)
    y = @. r * cos(φ_rad) * sin(λ_rad)
    z = @. r * sin(φ_rad)

    return x, y, z
end

Tℓx, Tℓy, Tℓz = location(Tfield)
Uℓx, Uℓy, Uℓz = location(ufield)
Vℓx, Vℓy, Vℓz = location(vfield)

λ = λnodes(grid.underlying_grid, Tℓx(), Tℓy(), Tℓz())
φ = φnodes(grid.underlying_grid, Tℓx(), Tℓy(), Tℓz())

Tx, Ty, Tz = spherical_coordinates_viz(λ, φ, 1)

λ = λnodes(grid.underlying_grid, Uℓx(), Uℓy(), Uℓz())
φ = φnodes(grid.underlying_grid, Uℓx(), Uℓy(), Uℓz())

Ux, Uy, Uz = spherical_coordinates_viz(λ, φ, 1)

λ = λnodes(grid.underlying_grid, Vℓx(), Vℓy(), Vℓz())
φ = φnodes(grid.underlying_grid, Vℓx(), Vℓy(), Vℓz())

Vx, Vy, Vz = spherical_coordinates_viz(λ, φ, 1)

earth_texture = load("../../figures/blue-marble-only-land.png")

n = 1024 ÷ 4 # 2048
lat = reverse(LinRange(-π/2, π/2, n))      # -π/2 = south pole, 0 = equator, π/2 = north pole
lon = LinRange(-π, π, 2*n)

r = 1.0
r_offset = 0.05   # 1% inflation

x = [ (r + r_offset) * cos(lat) * cos(lon) for lat in lat, lon in lon ]
y = [ (r + r_offset) * cos(lat) * sin(lon) for lat in lat, lon in lon ]
z = [ (r + r_offset) * sin(lat)         for lat in lat, lon in lon ]

# --- Extract depth levels (numbers before 'm') ---
depth_levels = [parse(Int, match(r"global_(\d+)", f).captures[1]) 
                for f in files_combined if occursin(r"global_\d+", f)]
unique_depth_levels = sort(unique(depth_levels))
depths_actual = abs.(grid.z.cᵃᵃᶠ[unique_depth_levels])

# --- Extract iteration numbers ---
iterations = [parse(Int, match(r"iteration(\d+)", f).captures[1]) 
              for f in files_combined if occursin(r"iteration\d+", f)]
unique_iterations = sort(unique(iterations))

vars = keys(jldopen(files_combined[1])["timeseries"])

files = sort(files_combined; rev=true)

# filepath = output_path *
#     "global_$(depth)_fields_$(resolution)_RYF_iteration$(iteration).jld2"


# f = jldopen(filepath)

# raw = f["$varpath/$(key)"]
####################################################################
function make_variable_video(var::String,
                             depths::Vector{Int},
                             depth_actual::Vector{Float64},
                             iterations::Vector{Int};
                             outname=nothing)

    is_speed = (var == "speed")

    all_depth_times = Vector{Vector{Float64}}()
    all_depth_data  = Vector{Vector{Matrix{Float32}}}()

    # --- pick depths properly (you hard-coded [75] in your snippet) ---
    for depth in [75]
        @info "Reading $depth m"

        # ---- pass 1: count total snapshots across iterations for this depth ----
        total_nt = 0
        for iteration in iterations
            filepath = output_path *
                "global_$(depth)_fields_$(resolution)_RYF_iteration$(iteration).jld2"
            f = jldopen(filepath)
            ts_keys = keys(f["timeseries/t"]) |> collect
            total_nt += length(ts_keys)
            close(f)
        end

        # ---- preallocate exactly ----
        raw_data  = Vector{Matrix{Float32}}(undef, total_nt)
        raw_times = Vector{Float64}(undef, total_nt)

        k = 0  # global frame counter across all iterations

        @time for iteration in iterations
            @show iteration
            filepath = output_path *
                "global_$(depth)_fields_$(resolution)_RYF_iteration$(iteration).jld2"

            # Open JLD2 once per file
            f = jldopen(filepath)

            # If you truly need sorted timesteps:
            ts_keys = sort!(parse.(Int, collect(keys(f["timeseries/t"]))))

            # Load field timeseries handles (these may open internally too)
            if is_speed
                has_u = FieldTimeSeries(filepath, "u")
                has_v = FieldTimeSeries(filepath, "v")
            else
                raw = FieldTimeSeries(filepath, var)
            end

            for (i, key) in enumerate(ts_keys)
                k += 1
                raw_times[k] = f["timeseries/t/$(key)"]

                if is_speed
                    # Pull numeric arrays and compute speed without constructing a new Field
                    u = interior(has_u[i])[:, :, 1]
                    v = interior(has_v[i])[:, :, 1]
                    A = @. sqrt(u*u + v*v)

                    # store a standalone array (avoid retaining views/references)
                    raw_data[k] = Array{Float32}(A)
                else
                    varpath = "timeseries/$var"
                    if !haskey(f, varpath)
                        @warn "Variable $var not found in $filepath. Skipping."
                        k -= 1
                        continue
                    end

                    A = interior(raw[i])[:, :, 1]
                    raw_data[k] = Array{Float32}(A)  # copy to detach from backing storage
                end
            end

            close(f)

            # If FieldTimeSeries supports closing, do it here.
            # (Some implementations don’t; harmless to omit if unsupported)
            # try close(has_u); close(has_v); catch; end
            # try close(raw); catch; end
        end

        # If any frames were skipped, truncate
        if k < total_nt
            raw_times = raw_times[1:k]
            raw_data  = raw_data[1:k]
        end

        # ---- sort times + data WITHOUT allocating giant new vectors ----
        order = sortperm(raw_times)
        permute!(raw_times, order)
        permute!(raw_data,  order)

        push!(all_depth_times, raw_times)
        push!(all_depth_data,  raw_data)
    end
    # return all_depth_times, all_depth_data
    nd = 1# length(depths)
    Nx, Ny = size(all_depth_data[1][1])

    # -------------------------------------------------------------------
    # Colormap & clim
    # -------------------------------------------------------------------
    if var == "S"
        clim = (32, 37)
        cmap = :haline
        cx, cy, cz = Tx, Ty, Tz
    elseif var  == "u"
        clim = (-0.5f0, 0.5f0)
        cmap = :bwr
        cx, cy, cz = Ux, Uy, Uz
    elseif var =="v"
        clim = (-0.5f0, 0.5f0)
        cmap = :bwr
        cx, cy, cz = Vx, Vy, Vz
    elseif var == "w"
        clim = (-0.00005f0, 0.00005f0)
        cmap = :bwr
        cx, cy, cz = Tx, Ty, Tz
    elseif var == "speed"
        clim = (0f0, 0.7f0)
        cmap = :Blues
        cx, cy, cz = Tx, Ty, Tz
    elseif var == "T"
        clim = (-2,35)
        cmap = :thermal
        cx, cy, cz = Tx, Ty, Tz
    end

    # -------------------------------------------------------------------
    # Build figure (2 × 3 grid)
    # -------------------------------------------------------------------
    fig = Figure(size=(750, 800))# colgap=0, rowgap=0, figure_padding=0)
    gl  = fig[1, 1] = GridLayout()

    # FIXED heights for UI rows

    land = (grid.immersed_boundary.bottom_height) .== 0
    land = view(land, :, :, 1)

    axs = Vector{Axis3}(undef, nd)
    hms = Vector{Surface}(undef, nd)

    # Preallocate observable matrices for each depth
    Z = [Observable(zeros(Float32, Nx, Ny)) for _ in 1:nd]
    for k in 1:nd
        #positions[k]
        land = (grid.immersed_boundary.bottom_height) .== 0
        land = view(land, :, :, 1)

        axs[k] = Axis3(gl[2,1], aspect=:data, viewmode = :fit)#, width = 300, height = 150)

        # earthvibes = surface!(axs[k], x, y, z;
        # color = earth_texture,
        # shading = NoShading,
        # backlight = 1.5f0
        # )

        hms[k] = surface!(
            axs[k],
            cx, cy, cz;  
            color=Z[k],                # <-- use observable
            colormap = cmap,
            nan_color = :darkgray,
            colorrange = clim
        )

        # earthvibes.rasterize = 3
        # hms[k].rasterize = 3

        hidedecorations!(axs[k])
        hidespines!(axs[k])

    end
    # Now row sizes work (rows exist)

    fig_title = Label(gl[1,1], "Loading...", tellwidth = false)
    # fig_title.position[] = Point2f(0.5, 0.98)
    Colorbar(gl[3,1], hms[1], label = "$var", vertical = false)
    # rowsize!(gl, 1, Auto(0.15))   # title
    # rowsize!(gl, 2, Relative(1))  # globe area
    # rowsize!(gl, 3, Auto(0.15))   # colorbar

    # colsize!(gl, 1, Relative(1))

    # colgap!(fig.layout, 1, Relative(-0.2))
    # resize_to_layout!(fig)

    # Output filename
    if isnothing(outname)
        outname = figdir * "$(var).mp4"
    end

    # -------------------------------------------------------------------
    # Animation: include year in title
    # -------------------------------------------------------------------

    times   = all_depth_times[1]
    years   = times ./ (365*24*60*60)
    nframes = length(times)
    @info "Preparing camera motion for $nframes frames..."
    eighth = Int(ceil(nframes / 8))

    h_lat_up1    = LinRange(-90, 0, eighth)
    h_lat_up2    = LinRange(0, 90, eighth)
    h_lat_down1  = LinRange(90, 0, eighth)
    h_lat_down2  = LinRange(0, -90, eighth)
    h_flat       = LinRange(0, 0, eighth)
    h_SO         = LinRange(-90, -90, eighth)
    h_arctic     = LinRange(90, 90, eighth)

    lat_sequence = vcat(h_flat, h_lat_down2, h_SO, h_lat_up1, h_flat, h_lat_up2, h_arctic, h_lat_down1)
    lat_sequence = lat_sequence[1:nframes]  # safety

    az_sequence = LinRange(-180, 180, nframes)

    # Choose which depth to animate (if you were looping over nd before)
    d = 1  # e.g. first depth
    @info "Recording video to $outname ..."

    t0 = time()

    Makie.record(fig, outname, 1:nframes; framerate=12) do frame
        elapsed = time() - t0
        rate = elapsed / frame
        remaining = rate * (nframes - frame)
        @info "Frame $frame/$nframes — ETA $(round(remaining/60, digits=1)) min"
        flush(stdout)
        az = az_sequence[frame]
        h_lat = lat_sequence[frame]
        
        fig_title.text[] = "Var: $var — Year = $(round(years[frame], digits=2))"
        for d in 1:nd
            Z[d][] = all_depth_data[d][frame] 
            Z[d][] = ifelse.(land, NaN32, Z[d][])  # mask land
            axs[d].elevation = deg2rad(h_lat)
            axs[d].azimuth   = deg2rad(az)
        end
    end

    @info "Saved → $outname"
    return nothing
end

####################################################################
# RUN
####################################################################

vars = ["T", "S", "speed",  "u", "v", "w"]

for var in vars
    @info "Processing $var..."
    make_variable_video(var, unique_depth_levels, depths_actual, unique_iterations;
                        outname = figdir * "$(var)_earth_vid.mp4")
end