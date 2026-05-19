using JLD2
using CairoMakie

files = [
    "RYF_sxtdeg_checkpoint_noSIatall_iteration144.jld2",
    "RYF_sxtdeg_checkpoint_noSIdyanamics_iteration144.jld2",
    "RYF_sxtdeg_checkpoint_noSIFW_iteration144.jld2",
    "RYF_sxtdeg_checkpoint_noSIHFFW_iteration144.jld2", 
    "RYF_sxtdeg_checkpoint_artefact_iteration144.jld2"
]

labels = [
    "No sea ice at all",
    "No sea ice dynamics",
    "No sea ice salinity flux",
    "No sea ice thermodynamics",
    "Coupled OSIM"
]

basepath = "/data/gpfs/projects/punim2499/taimoor/ocean-ensembles/outputs/"
outpath = "/data/gpfs/projects/punim2499/taimoor/ocean-ensembles/figures/"

w_list = []
valid_labels = []

for (file, label) in zip(files, labels)

    path = joinpath(basepath, file)

    if !isfile(path)
        @warn "Skipping missing file $path"
        continue
    end

    @info "Loading $file"

    jldopen(path, "r") do data
        w_temp = data["simulation/model/ocean/model/velocities/w/data"][1600:1800, 894:1094, 75]
        push!(w_list, w_temp)
        push!(valid_labels, label)
    end
end

fig = Figure(size = (200, 200 * length(w_list)))

for (i, (w, label)) in enumerate(zip(w_list, valid_labels))
    ax = Axis(fig[i, 1], title = label)

    heatmap!(
        ax,
        w,
        colorrange = (-5e-4, 5e-4),
        colormap = :balance
    )
    scatter!(ax, [1735-1600], [1085-894], color = :red)
end

save(joinpath(outpath, "w_plot_debugging.png"), fig, dpi=300)

@info "Saved figure to $(joinpath(outpath, "w_plot_debugging.png"))"