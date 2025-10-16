using CairoMakie

ngpus_v100 = [2, 3, 4, 8, 12]
time_per_100iterations = Array([90, 1.05*60, 55, 72, 70])

figure = Figure(size = (1200, 600))
ax1 = Axis(figure[1, 1]; xlabel = "Number of GPUs", ylabel = "Time per iteration (s)", title = "Scaling of 1/4-degree ClimaOcean simulation")
ax2 = Axis(figure[1, 2]; xlabel = "Number of GPUs", ylabel = "Total GPU walltime per iteration (s)")
scatterlines!(ax1, ngpus_v100, time_per_100iterations/100, color = :blue, label = "NVIDIA V100")
scatterlines!(ax2, ngpus_v100, time_per_100iterations/100 .* ngpus_v100, color = :blue)
lines!(ax2, ngpus_v100, time_per_100iterations[1]/100 * ngpus_v100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :black, label = "Ideal scaling")
ax1.xticks = (ngpus_v100, string.(ngpus_v100))
ax2.xticks = (ngpus_v100, string.(ngpus_v100))

axislegend(ax1, position = :rt)

figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

save(figdir * "scaling_qtr_deg_V100.png", figure)
