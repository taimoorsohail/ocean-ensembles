using CairoMakie
using Oceananigans  # From local
using Statistics
using JLD2

# output_path = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/outputs/")
# figdir = expanduser("/Users/tsohail/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/uom/ocean-ensembles-2/figures/")

output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved/")
figdir = expanduser("/g/data/v46/txs156/ocean-ensembles/figures/")

tot_files = ["global_3m_fields_onedeg_RYF_iteration0.jld2",
            "global_104m_fields_onedeg_RYF_iteration0.jld2",
            "global_507m_fields_onedeg_RYF_iteration0.jld2",
            "global_1027m_fields_onedeg_RYF_iteration0.jld2",
            "global_2038m_fields_onedeg_RYF_iteration0.jld2"]

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
