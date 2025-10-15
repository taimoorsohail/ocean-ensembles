using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2
using Glob
using OceanEnsembles

# output_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/outputs/")
# figdir = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/")
fig_avg = Figure(size = (1200, 800))

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved_fields/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

resolution = "sxtdeg"
nframes = nothing
# Example: get all matching files in a folder
files = glob("global_*$(resolution)_RYF_iteration*.jld2", output_path)

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
for prefix in unique_prefixes
    println("Combining files for prefix: $prefix")
    combine_ranks(prefix, prefix; remove_split_files = false, gridtype = "TripolarGrid")
end

files_combined = glob("global_*$(resolution)_RYF_iteration0.jld2", output_path)

# --- Extract depth levels (numbers before 'm') ---
depth_levels = [parse(Int, match(r"global_(\d+)m", f).captures[1]) 
                for f in files_combined if occursin(r"global_\d+m", f)]
unique_depth_levels = sort(unique(depth_levels))

# --- Extract iteration numbers ---
iterations = [parse(Int, match(r"iteration(\d+)", f).captures[1]) 
              for f in files_combined if occursin(r"iteration\d+", f)]
unique_iterations = sort(unique(iterations))

vars = ["T",
 "S",
 "u",
 "v",
 "w"]

# function create_dict(vars, path)
#     dicts = Dict()
#     for var in vars
#         try
#             # Surface
#             @info var
#             dicts[var] = FieldTimeSeries(path, var)
#         catch e
#             if e isa KeyError
#                 @warn "Skipping variable $var: Key not found in file."
#             else
#                 rethrow(e)
#             end
#         end
#     end
#     return dicts
# end

function create_dict(vars, path)
    results = asyncmap(vars; ntasks=100) do var
        try
            return (var => FieldTimeSeries(path, var))
        catch e
            if e isa KeyError
                @warn "Skipping variable $var: Key not found in file."
                return nothing
            else
                rethrow(e)
            end
        end
    end

    # filter out skipped ones
    results = filter(!isnothing, results)

    # turn into Dict
    return Dict(results)
end

avg_val = Dict(var => Dict() for var in vars)

pattern = "global_3m_*$(resolution)_RYF_iteration0.jld2"
matching_files = glob(pattern, output_path)
slice = create_dict(vars, matching_files[1])

grid = slice["T"].grid
area = Field{Center, Center, Nothing}(grid)
set!(area, 1)
area_2d = (Integral(area, dims = (1,2)) |> Field)[1,1,1]


ax1 = Axis(fig_avg[1, 1:3], title = "Temperature", xlabel = "Year", ylabel = "Average Temperature (°C)")
ax2 = Axis(fig_avg[1, 4:6], title = "Salinity", xlabel = "Year", ylabel = "Average Salinity (psu)")
ax3 = Axis(fig_avg[2, 1:2], title = "U velocity", xlabel = "Year", ylabel = "Average U (m/s)")
ax4 = Axis(fig_avg[2, 3:4], title = "V velocity", xlabel = "Year", ylabel = "Average V (m/s)")
ax5 = Axis(fig_avg[2, 5:6], title = "W velocity", xlabel = "Year", ylabel = "Average W (m/s)")

fig = Figure(size = (1200, 800))
fig_vel = Figure(size = (1200, 800))

# Observable for animation
frame_idx = Observable(1)


for (idx, depth) in enumerate(unique_depth_levels)   # ← your depth list
    # one dict per var for this depth, keyed by time
    merged = Dict(var => Dict() for var in vars)  # time => field
    for iteration_val in unique_iterations
        pattern = "global_$(depth)m_*$(resolution)_RYF_iteration$(iteration_val).jld2"
        matching_files = glob(pattern, output_path)

        if isempty(matching_files)
            @warn "No files found for depth: $depth m, iteration: $iteration_val"
            continue
        end

        @info "Processing depth: $depth m, iteration: $iteration_val"
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
    

    for var in vars
        @show var
        # Get sorted times

        sorted_times = sort(collect(keys(merged[var])))
        # preallocated_field = Field{Center, Center, Nothing}(merged[var][sorted_times[1]].grid)
        # Preallocate the list if you want (optional)
        # nested_list = Vector{Float64}(undef, length(sorted_times))
        nested_list = asyncmap(sorted_times; ntasks=100) do t
            field = merged[var][t]
            local_field = Field{Center, Center, Nothing}(grid)
            interior(local_field) .= field.data
            avg_field = (Integral(local_field, dims=(1,2)) |> Field)[1,1,1] / area_2d
            return avg_field
        end
        # for (i, t) in enumerate(sorted_times)
        #     @show i
        #     field = merged[var][t]
        #     interior(preallocated_field) .= field.data  # Copy data to preallocated field
        #     @time avg_field = (Integral(preallocated_field, dims = (1,2)) |> Field)[1,1,1] / area_2d
        #     nested_list[i] = avg_field
        # end

        avg_val[var][depth] = nested_list
        sorted_years = sorted_times ./ (3600 * 24 * 365)
        
        if var == "T"
            lines!(ax1, sorted_years, avg_val[var][depth], label = "$(depth)m")
            # ylims!(ax1, minimum(avg_val[var][depth]), maximum(avg_val[var][depth]))
            xlims!(ax1, 0, maximum(sorted_years))
        elseif var == "S"
            lines!(ax2, sorted_years, avg_val[var][depth], label = "$(depth)m")
            # ylims!(ax2, minimum(avg_val[var][depth]), maximum(avg_val[var][depth]))
            xlims!(ax2, 0, maximum(sorted_years))
        elseif var == "u"
            lines!(ax3, sorted_years, avg_val[var][depth], label = "$(depth)m")
            # ylims!(ax3, minimum(avg_val[var][depth]), maximum(avg_val[var][depth]))
            xlims!(ax3, 0, maximum(sorted_years))
        elseif var == "v"
            lines!(ax4, sorted_years, avg_val[var][depth], label = "$(depth)m")
            # ylims!(ax4, minimum(avg_val[var][depth]), maximum(avg_val[var][depth]))
            xlims!(ax4, 0, maximum(sorted_years))
        elseif var == "w"
            lines!(ax5, sorted_years, avg_val[var][depth], label = "$(depth)m")
            # ylims!(ax5, minimum(avg_val[var][depth]), maximum(avg_val[var][depth]))
            xlims!(ax5, 0, maximum(sorted_years))
        end

    end

    sorted_times = sort(collect(keys(merged["T"])))
    sorted_years = sorted_times ./ (3600 * 24 * 365)

    nframes = length(sorted_times)
    T = merged["T"]
    S = merged["S"]
    u = merged["u"]
    v = merged["v"]
    w = merged["w"]

    depth = T[sorted_times[1]].grid.z.cᵃᵃᶜ[first(T[sorted_times[1]].indices[3])]

    temp_data = @lift Array(dropdims(interior(T[sorted_times[$frame_idx]])-T[sorted_times[1]], dims=3))
    salt_data = @lift Array(dropdims(interior(S[sorted_times[$frame_idx]])-S[sorted_times[1]], dims=3))
    u_data = @lift Array(dropdims(interior(u[sorted_times[$frame_idx]])-u[sorted_times[1]], dims=3))
    v_data = @lift Array(dropdims(interior(v[sorted_times[$frame_idx]])-v[sorted_times[1]], dims=3))
    w_data = @lift Array(dropdims(interior(w[sorted_times[$frame_idx]])-w[sorted_times[1]], dims=3))

    ax_T = Axis(fig[idx, 1], title = "Depth = $(abs(round(depth, digits=1))) m")
    cax_T = fig[idx, 2]
    ax_S = Axis(fig[idx, 3], title = "Depth = $(abs(round(depth, digits=1))) m")
    cax_S = fig[idx, 4]
    ax_u = Axis(fig_vel[idx, 1], title = "Depth = $(abs(round(depth, digits=1))) m")
    cax_u = fig_vel[idx, 2]  
    ax_v = Axis(fig_vel[idx, 3], title = "Depth = $(abs(round(depth, digits=1))) m")
    cax_v = fig_vel[idx, 4]
    ax_w = Axis(fig_vel[idx, 5], title = "Depth = $(abs(round(depth, digits=1))) m")
    cax_w = fig_vel[idx, 6]
    hm = heatmap!(ax_T, temp_data; colormap=:bwr, colorrange=(-7.5,7.5))
    Colorbar(cax_T, hm, label="Temperature (°C)")
    hm = heatmap!(ax_S, salt_data; colormap=:bwr, colorrange=(-0.75,0.75))
    Colorbar(cax_S, hm, label="Salinity (g/kg)")
    hm = heatmap!(ax_u, u_data; colormap=:bwr, colorrange=(-.5,.5))
    Colorbar(cax_u, hm, label="u (m/s)")
    hm = heatmap!(ax_v, v_data; colormap=:bwr, colorrange=(-.5,.5))
    Colorbar(cax_v, hm, label="v (m/s)")
    hm = heatmap!(ax_w, w_data; colormap=:bwr, colorrange=(-.001,.001))
    Colorbar(cax_w, hm, label="w (m/s)")

    suptitle_text = @lift("Year = $(sorted_years[$frame_idx]) years")
    if idx == 1
        Label(fig[0, 1:4], suptitle_text, fontsize = 24, tellwidth = false, halign = :center)
        Label(fig_vel[0, 1:4], suptitle_text, fontsize = 24, tellwidth = false, halign = :center)

    end

end
# Record animation
record(fig, figdir * "slice_animation_tracer_$(resolution).mp4", 1:nframes; framerate = 20) do i
    frame_idx[] = i
end
record(fig_vel, figdir * "slice_animation_vel_$(resolution).mp4", 1:nframes; framerate = 20) do i
    frame_idx[] = i
end

# #### SURFACE PLOTS AND ANIMATION ####
# for depth in unique_depth_levels   # ← your depth list

# @info "I am loading $(depth)m" 
# slices_depth = []
# for tot_file in files
#     slice = create_dict(vars, output_path * tot_file)
#     push!(slices_depth, slice)
# end
# # Assume slice["T"] is your FieldTimeSeries
# T = slice["T"]
# S = slice["S"]
# u = slice["u"]
# v = slice["v"]
# w = slice["w"]

# depth = T.grid.z.cᵃᵃᶜ[first(T.indices[3])]

# # Observable for animation
# frame_idx = Observable(1)
# temp_data = @lift Array(dropdims(T[$frame_idx], dims=3))
# salt_data = @lift Array(dropdims(S[$frame_idx], dims=3))
# u_data = @lift Array(dropdims(u[$frame_idx], dims=3))
# v_data = @lift Array(dropdims(v[$frame_idx], dims=3))
# w_data = @lift Array(dropdims(w[$frame_idx], dims=3))

# fig = Figure(size = (1200, 800))
# ax = Axis(fig[1, 1])
# hm = heatmap!(ax, temp_data; colormap=:thermal, colorrange=(-2,35))
# Colorbar(fig[1, 2], hm, label="Temperature (°C)")
# ax = Axis(fig[1, 3])
# hm = heatmap!(ax, salt_data; colormap=:haline, colorrange=(35,37))
# Colorbar(fig[1, 4], hm, label="Salinity (g/kg)")
# ax = Axis(fig[2, 1])
# hm = heatmap!(ax, u_data; colormap=:bwr, colorrange=(-.5,.5))
# Colorbar(fig[2, 2], hm, label="u (m/s)")
# ax = Axis(fig[2, 3])
# hm = heatmap!(ax, v_data; colormap=:bwr, colorrange=(-.5,.5))
# Colorbar(fig[2, 4], hm, label="v (m/s)")
# ax = Axis(fig[3, 1])
# hm = heatmap!(ax, w_data; colormap=:bwr, colorrange=(-.001,.001))
# Colorbar(fig[3, 2], hm, label="w (m/s)")

# suptitle_text = @lift("Time = $($frame_idx) days, Depth = $(abs(round(depth, digits=1))) m")

# Label(fig[0, 1:4], suptitle_text, fontsize = 24, tellwidth = false, halign = :center)

# # Record animation
# record(fig, figdir * "slice_animation_$(abs(round(depth, digits=1))).mp4", 1:694; framerate = 20) do i
#     @info i
#     frame_idx[] = i
# end

# #### STANDARD DEVIATION ####

# T_array = Array(dropdims(T.data, dims=3))
# S_array = Array(dropdims(S.data, dims=3))
# U_array = Array(dropdims(u.data, dims=3))
# V_array = Array(dropdims(v.data, dims=3))
# W_array = Array(dropdims(w.data, dims=3))

# T_std = dropdims(mapslices(std, T_array; dims=3), dims=3)
# S_std = dropdims(mapslices(std, S_array; dims=3), dims=3)
# U_std = dropdims(mapslices(std, U_array; dims=3), dims=3)
# V_std = dropdims(mapslices(std, V_array; dims=3), dims=3)
# W_std = dropdims(mapslices(std, W_array; dims=3), dims=3)

# fig = Figure(size = (1200, 800))
# ax = Axis(fig[1, 1])
# hm = heatmap!(ax, T_std; colormap=:thermal, colorrange=(0,.5))
# Colorbar(fig[1, 2], hm, label="Temperature (°C)")
# ax = Axis(fig[1, 3])
# hm = heatmap!(ax, S_std; colormap=:haline, colorrange=(0,0.05))
# Colorbar(fig[1, 4], hm, label="Salinity (°C)")
# ax = Axis(fig[2, 1])
# hm = heatmap!(ax, U_std; colormap=:viridis, colorrange=(0,.05))
# Colorbar(fig[2, 2], hm, label="u (m/s)")
# ax = Axis(fig[2, 3])
# hm = heatmap!(ax, V_std; colormap=:viridis, colorrange=(0,.05))
# Colorbar(fig[2, 4], hm, label="v (m/s)")
# ax = Axis(fig[3, 1])
# hm = heatmap!(ax, W_std; colormap=:viridis, colorrange=(0,.0005))
# Colorbar(fig[3, 2], hm, label="w (m/s)")

# suptitle_text = "STD, Depth = $(abs(round(depth, digits=1))) m"

# Label(fig[0, 1:4], suptitle_text, fontsize = 24, tellwidth = false, halign = :center)

# save(figdir * "slice_std_$(abs(round(depth, digits=1))).png", fig, px_per_unit=3)=#

save(figdir * "average_slice_vars_$(resolution).png", fig_avg, px_per_unit=3)
