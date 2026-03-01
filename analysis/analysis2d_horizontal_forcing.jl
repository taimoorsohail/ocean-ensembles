using CairoMakie
using JLD2
using Glob

const output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved_fields/onedeg/")
const figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")
const resolution = "onedeg"
const SELECTED_VARS = [
    "total_heat_flux",
    "total_freshwater_flux",
    "total_freshwater_flux_with_salt_equiv",
]
const VAR_TITLES = Dict(
    "total_heat_flux" => "Total Heat Flux (W m⁻²)",
    "total_freshwater_flux" => "Mass Flux (kg m⁻² s⁻¹)",
    "total_freshwater_flux_with_salt_equiv" => "Mass Flux + Salt Equivalent (kg m⁻² s⁻¹)",
)

function run_id(path::AbstractString)
    m = match(r"run(\d+)", basename(path))
    return m === nothing ? -1 : parse(Int, m.captures[1])
end

function forcing_files(path::AbstractString)
    files = glob("global_forcing_fields_*_RYF_run*.jld2", path)
    files = filter(files) do f
        !occursin("_rank", f) && occursin("forcing_field", f) && run_id(f) >= 0
    end
    sort!(files; by = run_id)
    return files
end

function load_forcing_timeseries(files::Vector{String})
    isempty(files) && error("No forcing files found in $output_path.")

    vars = jldopen(files[1], "r") do f
        filter(k -> k != "t", collect(keys(f["timeseries"])))
    end

    all_data = Dict{String, Vector{Matrix{Float32}}}(v => Matrix{Float32}[] for v in vars)
    all_time = Dict{String, Vector{Float64}}(v => Float64[] for v in vars)

    for file in files
        @info "Reading $file"
        jldopen(file, "r") do f
            ts_keys = sort(parse.(Int, collect(keys(f["timeseries/t"]))))
            for key in ts_keys
                tval = Float64(f["timeseries/t/$key"])
                for var in vars
                    raw = f["timeseries/$var/$key"]
                    A = ndims(raw) == 3 ? Float32.(raw[:, :, 1]) : Float32.(raw)
                    push!(all_data[var], A)
                    push!(all_time[var], tval)
                end
            end
        end
    end

    for var in vars
        order = sortperm(all_time[var])
        all_time[var] = all_time[var][order]
        all_data[var] = all_data[var][order]
    end

    return vars, all_time, all_data
end

function colormap_and_limits(var::String, data::Vector{Matrix{Float32}})
    n = 0
    μ = 0.0
    m2 = 0.0

    for A in data
        for x in A
            n += 1
            δ = x - μ
            μ += δ / n
            m2 += δ * (x - μ)
        end
    end

    σ = n > 1 ? sqrt(m2 / (n - 1)) : 0.0
    σ = max(σ, eps(Float64))
    return :balance, (-σ, σ)
end

function make_forcing_animation(; outname = figdir * "forcing_fields_$(resolution)_all_runs.mp4", framerate = 6)
    files = forcing_files(output_path)
    @info "Using files:\n$(join(files, '\n'))"

    vars, all_time, all_data = load_forcing_timeseries(files)
    missing = setdiff(SELECTED_VARS, vars)
    isempty(missing) || error("Missing required flux variables: $(join(missing, \", \")). Available: $(join(vars, \", \"))")
    vars = SELECTED_VARS

    nframes = length(all_data[vars[1]])
    nframes == 0 && error("No timesteps found in forcing files.")

    fig = Figure(size = (1800, 700))
    title = Label(fig[0, :], "Loading...", tellwidth = false)

    observables = Dict{String, Observable{Matrix{Float32}}}()
    for (i, var) in enumerate(vars)
        ax = Axis(fig[1, i], title = get(VAR_TITLES, var, var))
        observables[var] = Observable(all_data[var][1])
        cmap, clim = colormap_and_limits(var, all_data[var])
        hm = heatmap!(ax, observables[var], colormap = cmap, colorrange = clim)
        Colorbar(fig[2, i], hm, vertical = false)
    end
    resize_to_layout!(fig)

    time = all_time[vars[1]]
    years = time ./ (365 * 24 * 60 * 60)

    record(fig, outname, 1:nframes; framerate) do frame
        title.text = "Global surface forcing fields (divergent scale, ±1σ) | Year = $(round(years[frame], digits=2))"
        for var in vars
            observables[var][] = all_data[var][frame]
        end
    end

    @info "Saved animation to $outname"
    return outname
end

make_forcing_animation()


if abspath(PROGRAM_FILE) == @__FILE__
    make_forcing_animation()
end
