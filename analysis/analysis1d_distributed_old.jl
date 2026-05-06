using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2
using Glob
using OceanEnsembles

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved_fields/old/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

keys_interest = ["T_totintegral",
                "S_totintegral"]

resolution = "sxtdeg"
nframes = nothing
# Example: get all matching files in a folder
files = glob("global_*tot*$(resolution)*_RYF_iteration*.jld2", output_path)
files_surface = filter(f -> !occursin("_rank", f),
                        glob("global_*forcing*$(resolution)_RYF_iteration*.jld2", output_path))

# Collect prefixes here
prefixes = String[]

for file in files
    fname = basename(file)
    prefix = replace(fname, r"_iteration.*" => "")
    push!(prefixes, joinpath(output_path, prefix))
end

# Keep only unique prefixes
unique_prefixes = unique(prefixes)

println(unique_prefixes)

iter_rank_map = identify_combination_targets(basename(unique_prefixes[1]), dirname(unique_prefixes[1]); type = "iterrank", iterrun = "iteration")
iterations = sort(collect(keys(iter_rank_map)))
ranks = iter_rank_map[iterations[1]]
time_total  = Vector{Vector{Float64}}()
iters_total = Vector{Vector{Int}}()
T_int_iters = Vector{Matrix{Float64}}()
S_int_iters = Vector{Matrix{Float64}}()

depth = []

# Read depth once from rank0 file of the first iteration
run0 = lpad(string(iterations[1]), 4, '0')
filename_rank0_first = string(unique_prefixes[1], "_iteration$(run0)_rank", ranks[1], ".jld2")
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
        "_iteration$(run)_rank",
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
            "_iteration$(run)_rank",
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
                end
            end
        end
    end

    push!(time_total, times)
    push!(iters_total, iters)
    push!(T_int_iters, T_int)
    push!(S_int_iters, S_int)
end

# Concatenate after loop
t_all = vcat(time_total...)
T_all = vcat(T_int_iters...)
S_all = vcat(S_int_iters...)


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


fig = Figure(size = (600, 600))
ax1 = Axis(fig[1, 1], title = "Ocean Heat Content", xlabel = "Time (years)", ylabel = "OHC (J)")
ax3 = Axis(fig[2, 1], title = "Ocean Freshwater Content", xlabel = "Time (years)", ylabel = "FW (m3)")

lines!(ax1, time_in_years, 1035*1000*T_all[:,1,1])
lines!(ax3, time_in_years, (1.3358605008598876e18.-S_all[:,1,1]./35))
xlims!(ax3, 0.5,12)
xlims!(ax1, 0.5,12)

save(figdir * "integrated_props_$(resolution).png", fig, px_per_unit=3)