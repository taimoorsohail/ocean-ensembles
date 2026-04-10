using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2
using Glob
using OceanEnsembles

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

keys_interest = ["T_totintegral",
                "S_totintegral",
                "T_vertintegral",
                "S_vertintegral",
                "total_volume_c",
                "vert_volume_c"]

resolution = "sxtdeg"
nframes = nothing
# Example: get all matching files in a folder
files = glob("global_*tot*$(resolution)*_RYF_run*.jld2", output_path)
files_surface = filter(f -> !occursin("_rank", f),
                        glob("global_*forcing*$(resolution)_RYF_run*.jld2", output_path))

# Collect prefixes here
prefixes = String[]

for file in files
    @info file
    fname = basename(file)
    prefix = replace(fname, r"_run.*" => "")
    push!(prefixes, joinpath(output_path, prefix))
end

# Keep only unique prefixes
unique_prefixes = unique(prefixes)

println(unique_prefixes)

iter_rank_map = identify_combination_targets(basename(unique_prefixes[1]), dirname(unique_prefixes[1]); type = "iterrank")
iterations = sort(collect(keys(iter_rank_map)))
ranks = iter_rank_map[iterations[1]]
time_total  = Vector{Vector{Float64}}()
iters_total = Vector{Vector{Int}}()
T_int_iters = Vector{Matrix{Float64}}()
S_int_iters = Vector{Matrix{Float64}}()
T_int_z_iters = Vector{Array{Float64,3}}()
S_int_z_iters = Vector{Array{Float64,3}}()
V_int_z_iters = Vector{Array{Float64,3}}()
V_int_iters = Vector{Matrix{Float64}}()

depth = []

# Read depth once from rank0 file of the first iteration
run0 = lpad(string(iterations[1]), 4, '0')
filename_rank0_first = string(unique_prefixes[1], "_run$(run0)_rank", ranks[1], ".jld2")
jldopen(filename_rank0_first, "r") do data
    push!(depth, data["grid/underlying_grid/z/cᵃᵃᶜ"][7:end-7] ) # this is your z coordinate
end

Lz = length(depth[1])


for iteration in iterations
    @info "Processing run $iteration"
    run = lpad(string(iteration), 4, '0')

    # Current global max time seen so far
    tmax = isempty(time_total) ? -Inf : maximum(maximum.(time_total))

    # ---- First pass: collect new times (rank 0 is enough) ----
    times = Float64[]
    iters = Int[]

    filename_rank0 = string(
        unique_prefixes[1],
        "_run$(run)_rank",
        ranks[1],
        ".jld2"
    )

    jldopen(filename_rank0, "r") do data
        timeiters = sort(parse.(Int, keys(data["timeseries/t/"])))
        for iter in timeiters
            t = data["timeseries/t/$(iter)"]
            if t > tmax
                push!(times, t)
                push!(iters, iter)
            end
        end
    end

    # Allocate correctly-sized array
    T_int = zeros(length(times), length(ranks))
    S_int = zeros(length(times), length(ranks))
    T_int_z = zeros(length(times), length(ranks), Lz)
    S_int_z = zeros(length(times), length(ranks), Lz)
    V_int = zeros(length(times), length(ranks))
    V_int_z = zeros(length(times), length(ranks), Lz)

    # ---- Second pass: fill T_int for all ranks ----
    for (j, rank) in enumerate(ranks)
        filename = string(
            unique_prefixes[1],
            "_run$(run)_rank",
            rank,
            ".jld2"
        )

        row = 0
        jldopen(filename, "r") do data
            timeiters = sort(parse.(Int, keys(data["timeseries/t/"])))
            for iter in timeiters
                t = data["timeseries/t/$(iter)"]
                if t > tmax
                    row += 1
                    T_int[row, j] =
                        data["timeseries/T_totintegral/$(iter)"][1, 1, 1]
                    S_int[row, j] =
                        data["timeseries/S_totintegral/$(iter)"][1, 1, 1]
                    T_int_z[row, j,:] =
                        data["timeseries/T_vertintegral/$(iter)"][1, 1, 7:end-7]
                    S_int_z[row, j,:] =
                        data["timeseries/S_vertintegral/$(iter)"][1, 1, 7:end-7]
                    V_int[row, j] =
                        data["timeseries/total_volume_c/$(iter)"][1, 1, 1]
                    V_int_z[row, j,:] =
                        data["timeseries/vert_volume_c/$(iter)"][1, 1, 7:end-7]
                end
            end
        end
    end

    push!(time_total, times)
    push!(iters_total, iters)
    push!(T_int_iters, T_int)
    push!(S_int_iters, S_int)
    push!(T_int_z_iters, T_int_z)
    push!(S_int_z_iters, S_int_z)
    push!(V_int_iters, V_int)
    push!(V_int_z_iters, V_int_z)
end

# Concatenate after loop
t_all = vcat(time_total...)
T_all = vcat(T_int_iters...)
S_all = vcat(S_int_iters...)
V_all = vcat(V_int_iters...)

T_z_all = vcat(T_int_z_iters...)
S_z_all = vcat(S_int_z_iters...)
V_z_all = vcat(V_int_z_iters...)


# time_total = []
# iters_total = []
# T_int_iters = []
# for iteration in iterations
#     data = jldopen(string(unique_prefixes[1], "_iteration$(iteration)_rank", ranks[1], ".jld2"))
#     timeiters = sort(parse.(Int, keys(data["timeseries/t/"])))
#     time = []
#     @info "Processing iteration $iteration"
#     for (i, iter) in enumerate(timeiters)
#         @info "Checking time for iter $iter"
#         if isempty(time_total) || data["timeseries/t/$(iter)"] > maximum(maximum.(time_total))
#             push!(time, data["timeseries/t/$(iter)"])
#         end
#     end
#     T_int = zeros(length(time), length(ranks))
#     for rank in ranks
#         data = jldopen(string(unique_prefixes[1], "_iteration$(iteration)_rank", rank, ".jld2"))
#         for (i, iter) in enumerate(timeiters)
#             if isempty(time_total) || data["timeseries/t/$(iter)"] > maximum(maximum.(time_total))
#                 T_int[i, rank+1] = data["timeseries/T_totintegral/$(iter)"][1,1,1]
#             end
#         end
#     end
#     @info "I made it" 
#     push!(time_total, time)
#     push!(iters_total, timeiters)
#     push!(T_int_iters, T_int)
# end

# T_all = vcat(T_int_iters...)
# t_all = vcat(time_total...)
# perm = sortperm(t_all)

# t_sorted = t_all[perm]
# T_sorted = T_all[perm, :]

time_in_years = t_all ./ (3600*24*365)


fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
ax2 = Axis(fig[2, 1], title = "Mean Temperature", xlabel = "Time (years)", ylabel = "Temperature (°C)")
ax3 = Axis(fig[1, 2], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
ax4 = Axis(fig[2, 2], title = "Mean Salinity", xlabel = "Time (years)", ylabel = "Salinity (psu)")
ax5 = Axis(fig[3, :], title = "Total Volume", xlabel = "Time (years)", ylabel = "Volume (m³)")

lines!(ax1, time_in_years, 1035*1000*sum(T_all, dims=2)[:,1], label = "OHC")
lines!(ax2, time_in_years, sum(T_all, dims=2)[:,1]./sum(V_all, dims=2)[:,1], label = "Mean Temperature")
lines!(ax3, time_in_years, sum(S_all, dims=2)[:,1]./(35*1035), label = "OSC")
lines!(ax4, time_in_years, sum(S_all, dims=2)[:,1]./sum(V_all, dims=2)[:,1], label = "Mean Salinity")
lines!(ax5, time_in_years, sum(V_all, dims=2)[:,1], label = "Total Volume")

save(figdir * "integrated_props_$(resolution).png", fig, px_per_unit=3)

fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
ax2 = Axis(fig[2, 1], title = "Mean Temperature", xlabel = "Time (years)", ylabel = "Temperature (°C)")
ax3 = Axis(fig[1, 2], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
ax4 = Axis(fig[2, 2], title = "Mean Salinity", xlabel = "Time (years)", ylabel = "Salinity (psu)")
ax5 = Axis(fig[3, :], title = "Total Volume", xlabel = "Time (years)", ylabel = "Volume (m³)")

allranks_T_z_int = sum(T_z_all, dims=2)[:,1,:]
allranks_T_z_int_anomaly = allranks_T_z_int .- allranks_T_z_int[1,:]'
allranks_S_z_int = sum(S_z_all, dims=2)[:,1,:]
allranks_S_z_int_anomaly = allranks_S_z_int .- allranks_S_z_int[1,:]'
allranks_V_z_int = sum(V_z_all, dims=2)[:,1,:]

heatmap!(ax1, time_in_years, depth[1], 1035*1000*allranks_T_z_int_anomaly, label = "OHC", colorrange = (-1e21, 1e21), colormap = :bwr)
heatmap!(ax2, time_in_years, depth[1], 1035*1000*allranks_T_z_int_anomaly./allranks_V_z_int, label = "Mean Temperature")
heatmap!(ax3, time_in_years, depth[1], allranks_S_z_int_anomaly./(35*1035), label = "OSC", colorrange = (-1e10, 1e10), colormap = :bwr)
heatmap!(ax4, time_in_years, depth[1], allranks_S_z_int_anomaly./allranks_V_z_int, label = "Mean Salinity")
heatmap!(ax5, time_in_years, depth[1], allranks_V_z_int, label = "Total Volume")

ylims!(ax1, -1000, 0)
ylims!(ax2, -1000, 0)
ylims!(ax3, -1000, 0)
ylims!(ax4, -1000, 0)
ylims!(ax5, -1000, 0)

save(figdir * "integrated_props_z_$(resolution).png", fig, px_per_unit=3)

# fig = Figure(size = (800, 600))
# ax1 = Axis(fig[1, 1], title = "OHC", xlabel = "Time (years)", ylabel = "OHC (J)")
# lines!(ax1, depth[1], 1035*1000*allranks_T_z_int_anomaly[16,:])
# xlims!(ax1, 0, -1000)
# ylims!(ax1, -1e20, 1e20)
# save(figdir * "test_temp_profile.png", fig, px_per_unit=3)