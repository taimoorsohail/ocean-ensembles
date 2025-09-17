using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2
using Glob

# output_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/outputs/")
# figdir = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/")

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

resolution = "onedeg"
# Example: get all matching files in a folder
files = glob("global_*$(resolution)_RYF_iteration*.jld2", output_path)

# --- Extract depth levels (numbers before 'm') ---
depth_levels = [parse(Int, match(r"global_(\d+)m", f).captures[1]) 
                for f in files if occursin(r"global_\d+m", f)]
unique_depth_levels = sort(unique(depth_levels))

# --- Extract iteration numbers ---
iterations = [parse(Int, match(r"iteration(\d+)", f).captures[1]) 
              for f in files if occursin(r"iteration\d+", f)]
unique_iterations = sort(unique(iterations))


vars = [ "T",
 "S",
 "u",
 "v",
 "w"]

function create_dict(vars, path)
    dicts = Dict()
    for var in vars
        try
            # Surface
            @info var
            dicts[var] = FieldTimeSeries(path, var)
        catch e
            if e isa KeyError
                @warn "Skipping variable $var: Key not found in file."
            else
                rethrow(e)
            end
        end
    end
    return dicts
end

avg_val = Dict(var => Dict() for var in vars)

for depth in unique_depth_levels   # ← your depth list
    # one dict per var for this depth, keyed by time
    merged = Dict(var => Dict() for var in vars)  # time => field
    for iteration in unique_iterations
        pattern = "global_$(depth)m_*iteration$(iteration).jld2"
        matching_files = glob(pattern, output_path)

        if isempty(matching_files)
            @warn "No files found for depth: $depth m, iteration: $iteration"
            continue
        end

        @info "Processing depth: $depth m, iteration: $iteration"
        slice = create_dict(vars, matching_files[1])
        for var in vars
            if haskey(slice, var)
                ft = slice[var]
                for (tind, t) in enumerate(ft.times)   # assuming FieldTimeSeries is iterable
                    merged[var][t] = ft[tind]   # overwrite if t already exists
                end
            end
        end
    end
    

    var = "T"
    # for var in vars
    #     @show var
    # Get sorted times

    sorted_times = sort(collect(keys(merged[var])))

    # Preallocate the list if you want (optional)
    nested_list = Vector{Float64}(undef, length(sorted_times))

        for (i, t) in enumerate(sorted_times)
            @show i
            field = merged[var][t]
            avg_field = Average(field)
            nested_list[i] = Field(avg_field)[1,1,1]
            @show (avg_field)
        end

    avg_val[var][depth] = nested_list
end
    # end
    sorted_years = sorted_times ./ (3600 * 24 * 365)
    
    fig = Figure(size = (1200, 800))
    # 1. Temperature
    ax1 = Axis(fig[1, 1:3], title = "Temperature", xlabel = "Year", ylabel = "Average Temperature (°C)")
    lines!(ax1, sorted_years, avg_val["T"]["3m"])
    # lines!(ax1, time_year2, filter(!isnan,T_avg2), label = "No Checkpoint")
    xlims!(ax1, 0, maximum(sorted_years)  )
    ylims!(ax1, minimum(avg_val["T"]), maximum(avg_val["T"]))

    save(figdir * "average_slice_vars_$(resolution).png", fig, px_per_unit=3)

    # # 2. Salinity
    # ax2 = Axis(fig[1, 4:6], title = "Salinity", xlabel = "Year", ylabel = "Average Salinity (psu)")
    # lines!(ax2, time_year, filter(!isnan,S_avg), label = "Checkpoint")
    # # lines!(ax2, time_year2, filter(!isnan,S_avg2), label = "No Checkpoint")
    # xlims!(ax2, 0, maximum(time_year))
    # ylims!(ax2, minimum(S_avg), maximum(S_avg))

    # # 3. U velocity
    # ax3 = Axis(fig[2, 1:2], title = "U velocity", xlabel = "Year", ylabel = "Average U (m/s)")
    # lines!(ax3, time_year, filter(!isnan,u_avg), label = "Checkpoint")
    # # lines!(ax3, time_year2, filter(!isnan,u_avg2), label = "No Checkpoint")
    # xlims!(ax3, 0, maximum(time_year))
    # ylims!(ax3, minimum(u_avg), maximum(u_avg))

    # # 4. V velocity
    # ax4 = Axis(fig[2, 3:4], title = "V velocity", xlabel = "Year", ylabel = "Average V (m/s)")
    # lines!(ax4, time_year, filter(!isnan,v_avg), label = "Checkpoint")
    # # lines!(ax4, time_year2, filter(!isnan,v_avg2), label = "No Checkpoint")
    # xlims!(ax4, 0, maximum(time_year))
    # ylims!(ax4, minimum(v_avg), maximum(v_avg))

    # # 5. W velocity
    # ax5 = Axis(fig[2, 5:6], title = "W velocity", xlabel = "Year", ylabel = "Average W (m/s)")
    # lines!(ax5, time_year, filter(!isnan,w_avg), label = "Checkpoint")
    # # lines!(ax5, time_year2, filter(!isnan,w_avg2), label = "No Checkpoint")
    # xlims!(ax5, 0, maximum(time_year))
    # ylims!(ax5, minimum(w_avg), maximum(w_avg))


#     # Convert back into time series objects (sorted by time)
#     depth_dict = Dict{String, Any}()
#     for var in vars
#         times = sort(collect(keys(merged[var])))
#         fields = [merged[var][t] for t in times]
#         depth_dict[var] = (times, fields)   # or wrap back into FieldTimeSeries if needed
#     end

#     push!(slices_depth_iter, depth_dict)
# # end

# for depth in unique_depth_levels
#     slices_iter = []
#     for iteration in unique_iterations
#         pattern = "global_$(depth)m_*iteration$(iteration).jld2"
#         matching_files = glob(pattern, output_path)
#         if !isempty(matching_files)
#             @info "Processing depth: $depth m, iteration: $iteration"
#             slice = create_dict(vars, matching_files[1])
#             push!(slices_iter, slice)
#         else
#             @warn "No files found for depth: $depth m, iteration: $iteration"
#         end
    
#     end
#     push!(slices_depth_iter, slices_iter)
# end

#=

@info "I am loading the surface" 
slices_depth = []
for tot_file in tot_files
    slice = create_dict(vars, output_path * tot_file)
    push!(slices_depth, slice)
end
# Assume slice["T"] is your FieldTimeSeries
T = slice["T"]
S = slice["S"]
u = slice["u"]
v = slice["v"]
w = slice["w"]

depth = T.grid.z.cᵃᵃᶜ[first(T.indices[3])]

# Observable for animation
frame_idx = Observable(1)
temp_data = @lift Array(dropdims(T[$frame_idx], dims=3))
salt_data = @lift Array(dropdims(S[$frame_idx], dims=3))
u_data = @lift Array(dropdims(u[$frame_idx], dims=3))
v_data = @lift Array(dropdims(v[$frame_idx], dims=3))
w_data = @lift Array(dropdims(w[$frame_idx], dims=3))

fig = Figure(size = (1200, 800))
ax = Axis(fig[1, 1])
hm = heatmap!(ax, temp_data; colormap=:thermal, colorrange=(-2,35))
Colorbar(fig[1, 2], hm, label="Temperature (°C)")
ax = Axis(fig[1, 3])
hm = heatmap!(ax, salt_data; colormap=:haline, colorrange=(35,37))
Colorbar(fig[1, 4], hm, label="Salinity (g/kg)")
ax = Axis(fig[2, 1])
hm = heatmap!(ax, u_data; colormap=:bwr, colorrange=(-.5,.5))
Colorbar(fig[2, 2], hm, label="u (m/s)")
ax = Axis(fig[2, 3])
hm = heatmap!(ax, v_data; colormap=:bwr, colorrange=(-.5,.5))
Colorbar(fig[2, 4], hm, label="v (m/s)")
ax = Axis(fig[3, 1])
hm = heatmap!(ax, w_data; colormap=:bwr, colorrange=(-.001,.001))
Colorbar(fig[3, 2], hm, label="w (m/s)")

suptitle_text = @lift("Time = $($frame_idx) days, Depth = $(abs(round(depth, digits=1))) m")

Label(fig[0, 1:4], suptitle_text, fontsize = 24, tellwidth = false, halign = :center)

# Record animation
record(fig, figdir * "slice_animation_$(abs(round(depth, digits=1))).mp4", 1:694; framerate = 20) do i
    @info i
    frame_idx[] = i
end

#### STANDARD DEVIATION ####

T_array = Array(dropdims(T.data, dims=3))
S_array = Array(dropdims(S.data, dims=3))
U_array = Array(dropdims(u.data, dims=3))
V_array = Array(dropdims(v.data, dims=3))
W_array = Array(dropdims(w.data, dims=3))

T_std = dropdims(mapslices(std, T_array; dims=3), dims=3)
S_std = dropdims(mapslices(std, S_array; dims=3), dims=3)
U_std = dropdims(mapslices(std, U_array; dims=3), dims=3)
V_std = dropdims(mapslices(std, V_array; dims=3), dims=3)
W_std = dropdims(mapslices(std, W_array; dims=3), dims=3)

fig = Figure(size = (1200, 800))
ax = Axis(fig[1, 1])
hm = heatmap!(ax, T_std; colormap=:thermal, colorrange=(0,.5))
Colorbar(fig[1, 2], hm, label="Temperature (°C)")
ax = Axis(fig[1, 3])
hm = heatmap!(ax, S_std; colormap=:haline, colorrange=(0,0.05))
Colorbar(fig[1, 4], hm, label="Salinity (°C)")
ax = Axis(fig[2, 1])
hm = heatmap!(ax, U_std; colormap=:viridis, colorrange=(0,.05))
Colorbar(fig[2, 2], hm, label="u (m/s)")
ax = Axis(fig[2, 3])
hm = heatmap!(ax, V_std; colormap=:viridis, colorrange=(0,.05))
Colorbar(fig[2, 4], hm, label="v (m/s)")
ax = Axis(fig[3, 1])
hm = heatmap!(ax, W_std; colormap=:viridis, colorrange=(0,.0005))
Colorbar(fig[3, 2], hm, label="w (m/s)")

suptitle_text = "STD, Depth = $(abs(round(depth, digits=1))) m"

Label(fig[0, 1:4], suptitle_text, fontsize = 24, tellwidth = false, halign = :center)

save(figdir * "slice_std_$(abs(round(depth, digits=1))).png", fig, px_per_unit=3)

# # Dimensions
# nx, ny, nt = size(T_array)

# # Compute skewness
# T_skew = [skewness(T_array[i,j,:]) for i in 1:nx, j in 1:ny]

# # Compute kurtosis
# T_kurt = [kurtosis(T_array[i,j,:]) for i in 1:nx, j in 1:ny]
=#