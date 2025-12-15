using CairoMakie
using JLD2
using Glob
using OceanEnsembles

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved_fields/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

resolution = "sxtdeg"

grid = create_grid(output_path * "global_75_fields_$(resolution)_RYF_iteration0", [0,1,2,3]; gridtype="TripolarGrid")

files_combined = filter(f -> !occursin("_rank", f),
                        glob("global_*$(resolution)_RYF_iteration*.jld2", output_path))

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

####################################################################
function make_variable_video(var::String,
                             depths::Vector{Int},
                             depth_actual::Vector{Float64},
                             iterations::Vector{Int};
                             outname=nothing)

    is_speed = (var == "speed")

    # -------------------------------------------------------------------
    # Load & sort time-series data
    # -------------------------------------------------------------------
    all_depth_times = Vector{Vector{Float64}}()
    all_depth_data  = Vector{Vector{Matrix{Float32}}}()

    for depth in depths
        raw_times = Float64[]
        raw_data  = Matrix{Float32}[]

        for iteration in iterations

            filepath = output_path *
                "global_$(depth)_fields_$(resolution)_RYF_iteration$(iteration).jld2"

            @info "Reading $filepath"

            f = jldopen(filepath)

            # timestep keys
            ts_keys = sort(parse.(Int, collect(keys(f["timeseries/t"]))))

            for key in ts_keys
                tval = f["timeseries/t/$(key)"]
                push!(raw_times, tval)
                if is_speed
                    # ---------------------------
                    # speed = sqrt(u^2 + v^2)
                    # ---------------------------
                    has_u = haskey(f, "timeseries/u")
                    has_v = haskey(f, "timeseries/v")

                    if !(has_u && has_v)
                        @warn "Missing u or v in $filepath (depth=$depth, iter=$iteration). Skipping timestep $key."
                        continue
                    end

                    raw_u = f["timeseries/u/$(key)"]
                    raw_v = f["timeseries/v/$(key)"]

                    if ndims(raw_u) < 2 || ndims(raw_v) < 2
                        @warn "u or v has unexpected dimensions at timestep $key, skipping."
                        continue
                    end

                    u = Float32.(ndims(raw_u) == 3 ? raw_u[:, :, 1] : raw_u)
                    v = Float32.(ndims(raw_v) == 3 ? raw_v[:, :, 1] : raw_v)

                    A = sqrt.(u.^2 .+ v.^2)

                else
                    # ---------------------------
                    # Normal variable: T, S, u, v, w, ...
                    # ---------------------------
                    varpath = "timeseries/$var"

                    if !haskey(f, varpath)
                        @warn "Variable $var not found in $filepath. Skipping."
                        continue
                    end

                    raw = f["$varpath/$(key)"]

                    # Determine correct slicing
                    if ndims(raw) == 2
                        A = Float32.(raw)
                    elseif ndims(raw) == 3
                        A = Float32.(raw[:, :, 1])
                    else
                        @warn "Unexpected shape for $var at timestep $key, skipping."
                        continue
                    end
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

    nd = length(depths)
    Nx, Ny = size(all_depth_data[1][1])

    # -------------------------------------------------------------------
    # Colormap & clim
    # -------------------------------------------------------------------
    if var == "S"
        clim = (34.8f0, 35.7f0)
        cmap = :viridis

    elseif var in ("u", "v", "w")
        clim = (-0.5f0, 0.5f0)
        cmap = :bwr

    elseif var == "speed"
        clim = (0f0, 0.7f0)
        cmap = :turbo

    else
        A0 = all_depth_data[1][end]
        clim = (minimum(A0), maximum(A0))
        cmap = :viridis
    end

    # -------------------------------------------------------------------
    # Build figure (2 × 3 grid)
    # -------------------------------------------------------------------
    fig = Figure(size = (1600, 900))
    fig.layout.widths = [1, 1, 1]   # Force all three columns equal
    fig.layout.heights = [1, 1]     # Optional: equal row height

    fig_title = fig[0, :] = Label(fig, "Loading...")

    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]

    axs = Vector{Axis}(undef, nd)
    hms = Vector{Heatmap}(undef, nd)

    # Preallocate observable matrices for each depth
    Z = [Observable(zeros(Float32, Nx, Ny)) for _ in 1:nd]

    for k in 1:nd
        (i, j) = positions[k]

        axs[k] = Axis(fig[i, j], title = "Depth $(depths_actual[k]) m")

        hms[k] = heatmap!(
            axs[k],
            Z[k];                  # <-- use observable
            colormap = cmap,
            colorrange = clim
        )

        Colorbar(fig, hms[k])
    end
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

    record(fig, outname, 1:nframes; framerate=1) do frame
        # fig_title.text = "Var: $var — Year = $(round(years[frame], digits=2))"

        for d in 1:nd
            Z[d][] = all_depth_data[d][frame]   # <-- observable update
        end
    end
    @info "Saved → $outname"
    return nothing
end

####################################################################
# RUN
####################################################################

vars = ["T", "S", "speed"]

for var in vars
    @info "Processing $var..."
    make_variable_video(var, unique_depth_levels, depths_actual, unique_iterations;
                        outname = figdir * "$(var)_all_depths.mp4")
end