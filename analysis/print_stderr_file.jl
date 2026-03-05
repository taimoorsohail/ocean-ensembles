using CairoMakie

# ================================
# 1. SET YOUR LOG FILE HERE
# ================================
output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/experiments/run_logs/")

logfiles = [
    output_path * "GPU_RYF1_6dg_h200_2.stderr",
    output_path * "GPU_RYF1_6dg_h200_3.stderr",
    output_path * "GPU_RYF1_6dg_h200_4.stderr"
]

# ==========================================
# 2. STORAGE (deduplicated by time)
# ==========================================
seen_time = Set{Float64}()

time_days = Float64[]

Tmax = Float64[]
Tmin = Float64[]

Smax = Float64[]
Smin = Float64[]

etamax = Float64[]
etamin = Float64[]

umax = Float64[]
vmax = Float64[]
wmax = Float64[]

# ==========================================
# 3. PARSE LOG FILES IN ORDER
# ==========================================
for logfile in logfiles
    @info "Parsing $logfile"

    for line in eachline(logfile)

        occursin("┌ Info:", line) || continue

        # ---- extract time ----
        tmatch = match(r"time:\s*([\d\.]+)\s*days", line)
        tmatch === nothing && continue
        t = parse(Float64, tmatch.captures[1])

        # ---- deduplicate by time ----
        if t in seen_time
            continue
        end
        push!(seen_time, t)
        push!(time_days, t)

        # ---- extrema(T) ----
        if (m = match(r"extrema\(T\):\s*\(([-\d\.eE+]+),\s*([-\d\.eE+]+)\)", line)) !== nothing
            push!(Tmax, parse(Float64, m.captures[1]))
            push!(Tmin, parse(Float64, m.captures[2]))
        end

        # ---- extrema(S) ----
        if (m = match(r"extrema\(S\):\s*\(([-\d\.eE+]+),\s*([-\d\.eE+]+)\)", line)) !== nothing
            push!(Smax, parse(Float64, m.captures[1]))
            push!(Smin, parse(Float64, m.captures[2]))
        end

        # ---- extrema(η) ----
        if (m = match(r"extrema\(η\):\s*\(([-\d\.eE+]+),\s*([-\d\.eE+]+)\)", line)) !== nothing
            push!(etamax, parse(Float64, m.captures[1]))
            push!(etamin, parse(Float64, m.captures[2]))
        end

        # ---- max|u| components ----
        if (m = match(r"max\|u\|:\s*\(([-\d\.eE+]+),\s*([-\d\.eE+]+),\s*([-\d\.eE+]+)\)", line)) !== nothing
            push!(umax, parse(Float64, m.captures[1]))
            push!(vmax, parse(Float64, m.captures[2]))
            push!(wmax, parse(Float64, m.captures[3]))
        end
    end
end

# ==========================================
# 4. SORT BY TIME (CRITICAL!)
# ==========================================
perm = sortperm(time_days)

time_days = time_days[perm]
Tmax  = Tmax[perm];   Tmin  = Tmin[perm]
Smax  = Smax[perm];   Smin  = Smin[perm]
etamax = etamax[perm]; etamin = etamin[perm]
umax  = umax[perm];   vmax  = vmax[perm];  wmax = wmax[perm]

@info "Parsed $(length(time_days)) unique timesteps from $(length(logfiles)) log files"

# ==========================================
# 5. PLOTTING
# ==========================================
fig = Figure(size = (1200, 900))

axT = Axis(fig[1,1], title="Temperature extrema", xlabel="Time (days)", ylabel="T (°C)")
lines!(axT, time_days, Tmax, label="T max")
axislegend(axT)

axS = Axis(fig[2,1], title="Salinity extrema", xlabel="Time (days)", ylabel="S (g/kg)")
lines!(axS, time_days, Smax, label="S max")
axislegend(axS)

axη = Axis(fig[3,1], title="Free surface extrema", xlabel="Time (days)", ylabel="η (m)")
lines!(axη, time_days, etamax, label="η max")
axislegend(axη)

axU = Axis(fig[4,1], title="Velocity maxima", xlabel="Time (days)", ylabel="m s⁻¹")
lines!(axU, time_days, umax, label="|u|max")
lines!(axU, time_days, vmax, label="|v|max")
axislegend(axU)

save("oceananigans_extrema_multifile.png", fig)
