using CairoMakie
using JLD2

fig = Figure(size = (3600,3600/2))
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

T_inits = []

for i in 0:3
    data = jldopen(output_path * "test_rank_stitching_rank$(i).jld2")
    iters = keys(data["timeseries/T"])
    T_init = data["timeseries/T/" * iters[2]]
    push!(T_inits, T_init)
    T_final = data["timeseries/T/" * iters[end]]

    ax_init = Axis(fig[1, i+1], title = "Temperature", xlabel = "Longitude", ylabel = "Latitude")
    ax_final = Axis(fig[2, i+1], title = "Temperature", xlabel = "Longitude", ylabel = "Latitude")

    hm2 = heatmap!(ax_init, view(T_init, :, :, 1); colormap = Reverse(:seismic), colorrange = (-90, 90))
    hm = heatmap!(ax_final, view(T_final, :, :, 1); colormap = Reverse(:seismic), colorrange = (-90, 90))

end

ax_init = Axis(fig[1, 1], title = "Temperature", xlabel = "Longitude", ylabel = "Latitude")

# Colorbar(fig[:,4], hm; label = "Temperature (°C)", vertical = true)
save(figdir * "Temperature_working.png", fig, px_per_unit=1)

# lines!(ax1, [755, 1010, 1010, 755, 755], [800,790, 920,920, 800])
# lines!(ax1, [679, 670, 679, 688, 679], [878-6,881-6, 888-6,881-6, 878-6])

