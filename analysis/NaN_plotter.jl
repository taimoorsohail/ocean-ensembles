using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2

output_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles/outputs/")
figdir = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles/figures/")

vars = ["ucpu", "vcpu", "Tcpu", "Scpu"]
NaNs = []
file = jldopen(output_path * "nan_state.jld2", "r")
for var in vars
    data = file[var]
    push!(NaNs, data)
end

close(file)

fig = Figure(size = (1600, 800))
# axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

title = string("Global 1 degree ocean simulation")

for (i, var) in enumerate(vars)
    row = Int(ceil(i / 2))
    col = 2 * ((i - 1) % 2) + 1  # 1 or 3 → leave space for colorbar

    axs = Axis(fig[row, col], xlabel = "Longitude (deg)", ylabel = "Latitude (deg)")
    hm = heatmap!(axs, view(interior(NaNs[i]), :, :, 50),
                  colormap = :bwr, nan_color = :lightgray)

    Colorbar(fig[row, col + 1], hm, label = string(var))
end

Label(fig[0, :], title, fontsize = 24)
save(figdir * "NaNs.png", fig, px_per_unit = 3)
