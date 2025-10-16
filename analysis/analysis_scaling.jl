using CairoMakie

ngpus_v100 = [2, 3, 4, 8, 12]
ngpus_a100 = [2]
time_per_100iterations_v100 = Array([90, 1.04*60, 55, 72, 70])
time_per_100iterations_a100 = Array([2.57*60])

figure = Figure(size = (1200, 600))
ax1 = Axis(figure[1, 1]; xlabel = "Number of GPUs", ylabel = "Time per iteration (s)", title = "Scaling of 1/4-degree ClimaOcean simulation")
ax2 = Axis(figure[1, 2]; xlabel = "Number of GPUs", ylabel = "Total GPU walltime per iteration (s)")
scatterlines!(ax1, ngpus_v100, time_per_100iterations_v100/100, color = :blue, label = "NVIDIA V100")
scatterlines!(ax2, ngpus_v100, time_per_100iterations_v100/100 .* ngpus_v100, color = :blue, label = "NVIDIA V100")
scatterlines!(ax1, ngpus_a100, time_per_100iterations_a100/100, color = :red, label = "NVIDIA A100")
scatterlines!(ax2, ngpus_a100, time_per_100iterations_a100/100 .* ngpus_a100, color = :red, label = "NVIDIA A100")

lines!(ax2, ngpus_v100, time_per_100iterations_v100[1]/100 * ngpus_v100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :black, label = "Ideal V100 scaling")
lines!(ax2, ngpus_a100, time_per_100iterations_a100[1]/100 * ngpus_a100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :grey, label = "Ideal A100 scaling")
ax1.xticks = (ngpus_v100, string.(ngpus_v100))
ax2.xticks = (ngpus_v100, string.(ngpus_v100))

axislegend(ax1, position = :rt)
axislegend(ax2, position = :lt)

figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

save(figdir * "scaling_qtr_deg.png", figure)
