using CairoMakie

ngpus_v100 = [2, 3, 4, 8, 12]
ngpus_v100_bindings = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

ngpus_a100 = [2]
# time_per_100iterations_v100 = Array([1.5*60, 1.04*60, 55, 72, 70])
# time_per_100iterations_v100_PR = Array([1.39*60, 55.870, 45.669, 53.068, 55.269])
# time_per_100iterations_v100_noCATKE = Array([1.02*60, 42.098, 39.561, 44.52, 44.947])
# time_per_100iterations_v100_latlon = Array([55, 37.328, 31.389, 41.266, 41.356])
# time_per_100iterations_v100_ocngns = Array([15.003, 10.499, 10.878, 11.506, 11.709])
# time_per_100iterations_v100_1_3 = Array([38.669, 28.520, 28.882, 31.158, 32.107])
time_per_100iterations_v100_bindings_1_3 = Array([39.9, 31.6, 21.5, 19.4, 23.7, 22.2, 18.7, 18.7, 21.5, 21.8, 18.2])
time_per_100iterations_v100_bindings_1_6 = Array([148.1, 106.2, 76.7, 63.2, 55.5, 49.4, 44.5, 40.5, 37.1, 37.3, 33.4])
time_per_100iterations_v100_bindings_1_6_seaice = Array([211.7, 196.9, 94.2, 115.7, 121.9, 135.4, 85.2, 86.4, 102.2, 122.3, 82.5])
time_per_100iterations_v100_bindings_1_6_CATKE_seaice = Array([261.2, 226.1, 120.6,  135.2, 137.8, 152.5, 97.5, 99.8, 112.8, NaN, 88.8])
time_per_100iterations_v100_bindings_1_6_CATKE = Array([185.2, 128.7, 93.7, 80.6, 68.9, 60.6, 52.1, 47.8, 48.6, 48.3, 43.2])

time_per_100iterations_a100 = Array([2.57*60])

figure = Figure(size = (1200, 600))
ax1 = Axis(figure[1, 1]; xlabel = "Number of GPUs", ylabel = "Simulated Years Per Day (SYPD)", title = "Scaling on Gadi NVIDIA V100 GPUs")
ax2 = Axis(figure[1, 2]; xlabel = "Number of GPUs", ylabel = "Scaling Efficiency (%)", title = "Scaling Efficiency on Gadi NVIDIA V100 GPUs")

# scatterlines!(ax1, ngpus_v100, time_per_100iterations_v100_PR/100, color = :green, label = "NVIDIA V100")
# scatterlines!(ax2, ngpus_v100, time_per_100iterations_v100_PR/100 .* ngpus_v100, color = :green, label = "NVIDIA V100")
# lines!(ax2, ngpus_v100, time_per_100iterations_v100_PR[1]/100 * ngpus_v100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :black)

# scatterlines!(ax1, ngpus_v100, time_per_100iterations_v100_noCATKE/100, color = :red, label = "ClimaOcean 1/4 tripolar")
# scatterlines!(ax2, ngpus_v100, time_per_100iterations_v100_noCATKE/100 .* ngpus_v100, color = :red, label = "ClimaOcean 1/4 tripolar")
# lines!(ax2, ngpus_v100, time_per_100iterations_v100_noCATKE[1]/100 * ngpus_v100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :black)

# scatterlines!(ax1, ngpus_v100, time_per_100iterations_v100_latlon/100, color = :blue, label = "ClimaOcean 1/4 lat-lon")
# scatterlines!(ax2, ngpus_v100, time_per_100iterations_v100_latlon/100 .* ngpus_v100, color = :blue, label = "ClimaOcean 1/4 lat-lon")
# lines!(ax2, ngpus_v100, time_per_100iterations_v100_latlon[1]/100 * ngpus_v100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :black)

# scatterlines!(ax1, ngpus_v100, time_per_100iterations_v100_1_3/100, color = :green, label = "ClimaOcean 1/3 lat-lon")
# scatterlines!(ax2, ngpus_v100, time_per_100iterations_v100_1_3/100 .* ngpus_v100, color = :green, label = "ClimaOcean 1/3 lat-lon")
# lines!(ax2, ngpus_v100, time_per_100iterations_v100_1_3[1]/100 * ngpus_v100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :black)

# scatterlines!(ax1, ngpus_v100, time_per_100iterations_v100_ocngns/100, color = :black, label = "Oceananigans 1/3 tripolar")
# scatterlines!(ax2, ngpus_v100, time_per_100iterations_v100_ocngns/100 .* ngpus_v100, color = :black, label = "Oceananigans 1/3 tripolar")
# lines!(ax2, ngpus_v100, time_per_100iterations_v100_ocngns[1]/100 * ngpus_v100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :black)

scatterlines!(ax1, ngpus_v100_bindings, 1 ./(36.5 .*time_per_100iterations_v100_bindings_1_3/100), color = :black, label = "ClimaOcean 1/3-deg")
scatterlines!(ax2, ngpus_v100_bindings, (time_per_100iterations_v100_bindings_1_3[1]/100 * ngpus_v100_bindings[1] .* ones(length(ngpus_v100_bindings)))./
                                        (time_per_100iterations_v100_bindings_1_3/100 .* ngpus_v100_bindings)*100, color = :black, label = "ClimaOcean 1/3-deg")

scatterlines!(ax1, ngpus_v100_bindings, 1 ./(36.5 .*time_per_100iterations_v100_bindings_1_6/100), color = :blue, label = "ClimaOcean 1/6-deg")
scatterlines!(ax2, ngpus_v100_bindings, (time_per_100iterations_v100_bindings_1_6[1]/100 * ngpus_v100_bindings[1] .* ones(length(ngpus_v100_bindings)))./
                                        (time_per_100iterations_v100_bindings_1_6/100 .* ngpus_v100_bindings)*100, color = :blue, label = "ClimaOcean 1/6-deg")

scatterlines!(ax1, ngpus_v100_bindings, 1 ./(36.5 .*time_per_100iterations_v100_bindings_1_6_seaice/100), color = :purple, label = "ClimaOcean 1/6-deg + sea ice")
scatterlines!(ax2, ngpus_v100_bindings, (time_per_100iterations_v100_bindings_1_6_seaice[1]/100 * ngpus_v100_bindings[1] .* ones(length(ngpus_v100_bindings)))./
                                        (time_per_100iterations_v100_bindings_1_6_seaice/100 .* ngpus_v100_bindings)*100, color = :purple, label = "ClimaOcean 1/6-deg + sea ice")

scatterlines!(ax1, ngpus_v100_bindings, 1 ./(36.5 .*time_per_100iterations_v100_bindings_1_6_CATKE_seaice/100), color = :red, label = "ClimaOcean 1/6-deg + seaice + CATKE")
scatterlines!(ax2, ngpus_v100_bindings, (time_per_100iterations_v100_bindings_1_6_CATKE_seaice[1]/100 * ngpus_v100_bindings[1] .* ones(length(ngpus_v100_bindings)))./
                                        (time_per_100iterations_v100_bindings_1_6_CATKE_seaice/100 .* ngpus_v100_bindings)*100, color = :red, label = "ClimaOcean 1/6-deg + seaice + CATKE")

scatterlines!(ax1, ngpus_v100_bindings, 1 ./(36.5 .*time_per_100iterations_v100_bindings_1_6_CATKE/100), color = :green, label = "ClimaOcean 1/6-deg + CATKE")
scatterlines!(ax2, ngpus_v100_bindings, (time_per_100iterations_v100_bindings_1_6_CATKE[1]/100 * ngpus_v100_bindings[1] .* ones(length(ngpus_v100_bindings)))./
                                        (time_per_100iterations_v100_bindings_1_6_CATKE/100 .* ngpus_v100_bindings)*100, color = :green, label = "ClimaOcean 1/6-deg + CATKE")

scatter!(ax1, 4, 1 ./((365/600)*20), color = :red, markersize = 10, label = "1/6-deg RYF Scaling Result")

# lines!(ax2, ngpus_a100, time_per_100iterations_a100[1]/100 * ngpus_a100[1] .* ones(length(ngpus_v100)); linestyle = :dash, color = :grey, label = "Ideal A100 scaling")
ax1.xticks = (ngpus_v100_bindings, string.(ngpus_v100_bindings))
ax2.xticks = (ngpus_v100_bindings, string.(ngpus_v100_bindings))

# axislegend(ax1, position = :rt)
axislegend(ax2, position = :rt)

figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

save(figdir * "scaling_qtr_deg.png", figure)
