using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2

output_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/outputs/")
figdir = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/")

var = "e"

NaN_field = FieldTimeSeries(output_path * "NaN_check_" * string(var) *".jld2", string(var))

fig = Figure(size = (1600, 800))
# axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

time_slice = lastindex(NaN_field.times)

times = NaN_field.times

title = string("Global 1 degree ocean simulation after ",
                         prettytime(times[time_slice] - times[1]))


axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
hm = heatmap!(axs, view(interior(NaN_field[time_slice]), :, :, 1), colorrange = (-1e-4,1e-4), colormap = :bwr, nan_color=:lightgray)
Colorbar(fig[1, 2], hm, label = string(var) * "(J)")
Label(fig[0, :], title)

save(figdir * "NaN_in_" * string(var) * ".png", fig, px_per_unit=3)