using ClimaOcean
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
r_offset = 0.001   # 1% inflation

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

    for depth in [75]
        @info "Reading  $depth m"
        raw_times = Float64[]
        raw_data  = Matrix{Float32}[]

        for iteration in iterations
            @show iteration
            filepath = output_path *
                "global_$(depth)_fields_$(resolution)_RYF_iteration$(iteration).jld2"

            if is_speed
                    # ---------------------------
                    # speed = sqrt(u^2 + v^2)
                    # ---------------------------
                    has_u = FieldTimeSeries(filepath, "u")
                    has_v = FieldTimeSeries(filepath, "v")
            else
                    raw = FieldTimeSeries(filepath, "$var")
            end

            f = jldopen(filepath)

            # timestep keys
            ts_keys = sort(parse.(Int, collect(keys(f["timeseries/t"]))))

            for (i, key) in enumerate(ts_keys)
                tval = f["timeseries/t/$(key)"]
                push!(raw_times, tval)
                if is_speed

                    raw_u = has_u[i]
                    raw_v = has_v[i]

                    A = @at (Center, Center, Nothing) sqrt(raw_u^2 + raw_v^2) |> Field
                    A = interior(A)[:, :, 1]  # extract 2D slice

                else
                    # ---------------------------
                    # Normal variable: T, S, u, v, w, ...
                    # ---------------------------
                    varpath = "timeseries/$var"

                    if !haskey(f, varpath)
                        @warn "Variable $var not found in $filepath. Skipping."
                        continue
                    end
                    A = interior(raw[i])[:, :, 1]

                end

                push!(raw_data, A)
            end

            close(f)
        end

        # Sort times + data
        order = sortperm(raw_times)
        push!(all_depth_times, raw_times[order])
        push!(all_depth_data,  raw_data[order])
    end

    nd = 1# length(depths)
    Nx, Ny = size(all_depth_data[1][1])

    # -------------------------------------------------------------------
    # Colormap & clim
    # -------------------------------------------------------------------
    if var == "S"
        clim = (34.8f0, 35.7f0)
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
    else
        A0 = all_depth_data[1][end]
        clim = (minimum(A0), maximum(A0))
        cmap = :thermal
        cx, cy, cz = Tx, Ty, Tz
    end

    # -------------------------------------------------------------------
    # Build figure (2 × 3 grid)
    # -------------------------------------------------------------------
    fig = Figure(
    colgap = 0,
    rowgap = 0,
    size = (800,800))

    land = (grid.immersed_boundary.bottom_height) .≥ 0
    land = view(land, :, :, 1)
    gl = fig[1, 1] = GridLayout()

    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]

    axs = Vector{Axis3}(undef, nd)
    hms = Vector{Surface}(undef, nd)

    # Preallocate observable matrices for each depth
    Z = [Observable(zeros(Float32, Nx, Ny)) for _ in 1:nd]
    for k in 1:nd
        (i, j) = (1,1)#positions[k]
        Z[k][][land] .= NaN

        axs[k] = Axis3(gl[i, j], aspect=:data, viewmode = :fit, protrusions = 0)#, width = 300, height = 150)

        hms[k] = surface!(
            axs[k],
            cx, cy, cz;  
            color=Z[k],                # <-- use observable
            colormap = cmap,
            colorrange = clim
        )

        surface!(axs[k], x, y, z;
        color = earth_texture,
        shading = NoShading,
        backlight = 1.5f0
        )

        hidedecorations!(axs[k])
        hidespines!(axs[k])

    end

    fig_title = Label(fig[0, :], "Loading...", tellwidth = false)
    Colorbar(gl[2,1], hms[1], label = "$var", vertical = false)

    # colgap!(fig.layout, 1, Relative(-0.2))
    resize_to_layout!(fig)

    # Output filename
    if isnothing(outname)
        outname = figdir * "$(var).mp4"
    end

    # -------------------------------------------------------------------
    # Animation: include year in title
    # -------------------------------------------------------------------
    times = all_depth_times[1]
    years = times ./ (365*24*60*60)
    nframes = length(times)

    record(fig, outname, 1:nframes; framerate=3) do frame
        fig_title.text = "Var: $var — Year = $(round(years[frame], digits=2))"
        half_clg = Int(ceil((nframes - 6)/2.0))
        qtr_clg = Int(ceil((nframes - 6)/4))
        eighth_clg = Int(ceil((nframes - 6)/8))

        # Precompute the two main halves
        h_lat_up1   = LinRange(-90, 0, eighth_clg)
        h_lat_up2 = LinRange(0, 90, eighth_clg)[2:end]
        h_lat_down1   = LinRange(90, 0, eighth_clg)
        h_lat_down2 = LinRange(0, -90, eighth_clg)[2:end]
        h_flat = LinRange(0, 0, qtr_clg)[2:end]
        # Full sequence with 3-frame tail from the start
        lat_sequence = vcat(h_lat_down2[end-2:end], h_lat_up1, h_flat, h_lat_up2, h_lat_down1, h_flat, h_lat_down2, h_lat_up1[1:3])
        # Latitude for current frame
        h_lat = lat_sequence[frame]
        
        # Step size for the sweep
        start_az = -190
        end_az   = 190

        # Main sweep
        az_main = LinRange(start_az, end_az, nframes)

        # Full azimuth sequence
        az_sequence = az_main
        az = az_sequence[frame]

        for d in 1:nd
            Z[d][] = all_depth_data[d][frame] 
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