using CairoMakie
using JLD2
using Glob
using OceanEnsembles
using Oceananigans: @at
using Oceananigans

const output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved_fields/onedeg/")
const figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")
const resolution = "onedeg"

grid = create_grid(output_path * "global_75_fields_$(resolution)_RYF_run0001"; gridtype = "TripolarGrid")

files_combined = filter(f -> !occursin("_rank", f),
                        glob("global_*$(resolution)_RYF_run*.jld2", output_path))

depth_levels = [parse(Int, match(r"global_(\d+)", f).captures[1])
                for f in files_combined if occursin(r"global_\d+", f)]
unique_depth_levels = sort(unique(depth_levels))
depths_actual = abs.(grid.z.cᵃᵃᶠ[unique_depth_levels])

iterations = [parse(Int, match(r"run(\d+)", f).captures[1])
              for f in files_combined if occursin(r"run\d+", f)]
unique_iterations = sort(unique(iterations))

function make_variable_video(var::String,
                             depths::Vector{Int},
                             depth_actual::Vector{Float64},
                             iterations::Vector{Int};
                             outname = nothing)
    is_speed = (var == "speed")

    all_depth_times = Vector{Vector{Float64}}()
    all_depth_data = Vector{Vector{Matrix{Float32}}}()

    for depth in depths
        @info "Reading $depth m"
        raw_times = Float64[]
        raw_data = Matrix{Float32}[]

        for iteration in iterations
            run = lpad(string(iteration), 4, '0')
            filepath = output_path * "global_$(depth)_fields_$(resolution)_RYF_run$(run).jld2"
            isfile(filepath) || continue

            if is_speed
                has_u = FieldTimeSeries(filepath, "u")
                has_v = FieldTimeSeries(filepath, "v")
            end

            f = jldopen(filepath, "r")
            ts_keys = sort(parse.(Int, collect(keys(f["timeseries/t"]))))

            for (i, key) in enumerate(ts_keys)
                tval = f["timeseries/t/$(key)"]
                push!(raw_times, tval)

                if is_speed
                    raw_u = has_u[i]
                    raw_v = has_v[i]
                    A = @at (Center, Center, Nothing) sqrt(raw_u^2 + raw_v^2) |> Field
                    A = interior(A)[:, :, 1]
                else
                    varpath = "timeseries/$var"
                    if !haskey(f, varpath)
                        @warn "Variable $var not found in $filepath. Skipping."
                        continue
                    end

                    raw = f["$varpath/$(key)"]
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

        order = sortperm(raw_times)
        push!(all_depth_times, raw_times[order])
        push!(all_depth_data, raw_data[order])
    end

    nd = length(depths)
    Nx, Ny = size(all_depth_data[1][1])

    if var == "S"
        clim = (34.8f0, 35.7f0)
        cmap = :viridis
    elseif var in ("u", "v", "w")
        clim = (-0.5f0, 0.5f0)
        cmap = :bwr
    elseif var == "speed"
        clim = (0f0, 0.7f0)
        cmap = :speed
    else
        A0 = all_depth_data[1][end]
        clim = (minimum(A0), maximum(A0))
        cmap = :viridis
    end

    ncols = min(3, nd)
    nrows = cld(nd, ncols)
    fig = Figure(size = (550 * ncols, 320 * nrows + 120))
    axs = Vector{Axis}(undef, nd)
    hms = Vector{Heatmap}(undef, nd)
    Z = [Observable(zeros(Float32, Nx, Ny)) for _ in 1:nd]

    for k in 1:nd
        i = cld(k, ncols)
        j = (k - 1) % ncols + 1
        axs[k] = Axis(fig[i, j], title = "Depth $(round(depth_actual[k], digits=1)) m")
        hms[k] = heatmap!(axs[k], Z[k]; colormap = cmap, colorrange = clim)
    end

    fig_title = Label(fig[0, :], "Loading...", tellwidth = false)
    Colorbar(fig[nrows + 1, :], hms[1], label = "$var", vertical = false)
    resize_to_layout!(fig)

    if isnothing(outname)
        outname = figdir * "$(var)_$(resolution).mp4"
    end

    times = all_depth_times[1]
    years = times ./ (365 * 24 * 60 * 60)
    nframes = length(times)

    record(fig, outname, 1:nframes; framerate = 6) do frame
        fig_title.text = "Var: $var | Year = $(round(years[frame], digits=2))"
        for d in 1:nd
            Z[d][] = all_depth_data[d][frame]
        end
    end

    @info "Saved -> $outname"
    return nothing
end

function run_old_horizontal_benchmark()
    vars = ["T", "S", "u", "v", "speed"]
    elapsed = @elapsed begin
        for var in vars
            @info "Processing $var..."
            make_variable_video(var, unique_depth_levels, depths_actual, unique_iterations;
                                outname = figdir * "$(var)_$(resolution)_all_depths.mp4")
        end
    end

    @info "Old benchmark finished." elapsed_seconds = round(elapsed, digits = 3)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_old_horizontal_benchmark()
end
