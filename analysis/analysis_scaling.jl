using CairoMakie

ngpus_v100 = [2, 3, 4, 8, 12]
ngpus_a100 = [2]
time_per_100iterations_v100 = Array([1.5*60, 1.04*60, 55, 72, 70])
time_per_100iterations_v100_PR = Array([1.39*60, 55.870, 45.669, 53.068, 55.269])
time_per_100iterations_v100_noCATKE = Array([1.02*60, 42.098, 39.561, 44.52, 44.947])
time_per_100iterations_v100_latlon = Array([55, 37.328, 31.389, 41.266, 41.356])
time_per_100iterations_v100_ocngns = Array([15.003, 10.499, 10.878, 11.506, 11.709])
time_per_100iterations_v100_1_3 = Array([38.669, 28.520, 28.882, 31.158, 32.107])

time_per_100iterations_a100 = Array([2.57*60])

figure = Figure(size = (1200, 600))
ax1 = Axis(figure[1, 1]; xlabel = "Number of GPUs", ylabel = "Time per iteration (s)", title = "Scaling on Gadi NVIDIA V100 GPUs")
ax2 = Axis(figure[1, 2]; xlabel = "Number of GPUs", ylabel = "Total GPU walltime per iteration (s)")

# scatterlines!(ax1, ngpus_v100, time_per_100iterations_v100_PR/100, color = :green, label = "NVIDIA V100")
# scatterlines!(ax2, ngpus_v100, time_per_100iterations_v100_PR/100 .* ngpus_v100, color = :green, label = "NVIDIA V100")
# lines!(ax2, ngpus_v100, time_per_100iterations_v100_PR[1]/100 * ngpus_v100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :black)

scatterlines!(ax1, ngpus_v100, time_per_100iterations_v100_noCATKE/100, color = :red, label = "ClimaOcean 1/4 tripolar")
scatterlines!(ax2, ngpus_v100, time_per_100iterations_v100_noCATKE/100 .* ngpus_v100, color = :red, label = "ClimaOcean 1/4 tripolar")
lines!(ax2, ngpus_v100, time_per_100iterations_v100_noCATKE[1]/100 * ngpus_v100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :black)

scatterlines!(ax1, ngpus_v100, time_per_100iterations_v100_latlon/100, color = :blue, label = "ClimaOcean 1/4 lat-lon")
scatterlines!(ax2, ngpus_v100, time_per_100iterations_v100_latlon/100 .* ngpus_v100, color = :blue, label = "ClimaOcean 1/4 lat-lon")
lines!(ax2, ngpus_v100, time_per_100iterations_v100_latlon[1]/100 * ngpus_v100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :black)

scatterlines!(ax1, ngpus_v100, time_per_100iterations_v100_1_3/100, color = :green, label = "ClimaOcean 1/3 lat-lon")
scatterlines!(ax2, ngpus_v100, time_per_100iterations_v100_1_3/100 .* ngpus_v100, color = :green, label = "ClimaOcean 1/3 lat-lon")
lines!(ax2, ngpus_v100, time_per_100iterations_v100_1_3[1]/100 * ngpus_v100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :black)

scatterlines!(ax1, ngpus_v100, time_per_100iterations_v100_ocngns/100, color = :black, label = "Oceananigans 1/3 tripolar")
scatterlines!(ax2, ngpus_v100, time_per_100iterations_v100_ocngns/100 .* ngpus_v100, color = :black, label = "Oceananigans 1/3 tripolar")
lines!(ax2, ngpus_v100, time_per_100iterations_v100_ocngns[1]/100 * ngpus_v100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :black)


# lines!(ax2, ngpus_a100, time_per_100iterations_a100[1]/100 * ngpus_a100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :grey, label = "Ideal A100 scaling")
ax1.xticks = (ngpus_v100, string.(ngpus_v100))
ax2.xticks = (ngpus_v100, string.(ngpus_v100))

axislegend(ax1, position = :rt)
axislegend(ax2, position = :lt)

figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

save(figdir * "scaling_qtr_deg.png", figure)
