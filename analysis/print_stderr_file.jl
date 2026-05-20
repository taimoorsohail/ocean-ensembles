using CairoMakie

logfiles = [
    joinpath(@__DIR__, "..", "experiments", "run_logs", "GPU_RYF1_6dg_h200_1.stderr"),
    joinpath(@__DIR__, "..", "experiments", "run_logs", "GPU_RYF1_6dg_h200_2.stderr"),
]

outputs_dir = joinpath(@__DIR__, "..", "outputs")
output_figure = joinpath(@__DIR__, "GPU_RYF1_6dg_h200_extrema_with_checkpoints.png")

function parse_float_maybe(str::AbstractString)
    s = strip(str)
    try
        return parse(Float64, s)
    catch
        return NaN
    end
end

function parse_time_days(str::AbstractString, unit::AbstractString)
    t = parse_float_maybe(str)
    isnan(t) && return NaN

    if startswith(unit, "second")
        return t / (3600 * 24)
    elseif startswith(unit, "hour")
        return t / 24
    else
        return t
    end
end

function checkpoint_iteration_to_time_days(target_iter::Int, iterations::Vector{Int}, time_days::Vector{Float64})
    n = length(iterations)
    n == 0 && return NaN
    n == 1 && return iterations[1] == target_iter ? time_days[1] : NaN

    idx = searchsortedfirst(iterations, target_iter)

    if idx <= n && iterations[idx] == target_iter
        return time_days[idx]
    end

    if idx == 1
        i1, i2 = 1, 2
    elseif idx > n
        i1, i2 = n - 1, n
    else
        i1, i2 = idx - 1, idx
    end

    it1, it2 = iterations[i1], iterations[i2]
    t1, t2 = time_days[i1], time_days[i2]
    it1 == it2 && return NaN

    return t1 + (target_iter - it1) * (t2 - t1) / (it2 - it1)
end

function checkpoint_iterations_from_outputs(outputs_dir::AbstractString)
    checkpoint_iters = Int[]

    if !isdir(outputs_dir)
        @warn "Outputs directory not found: $outputs_dir"
        return checkpoint_iters
    end

    for name in readdir(outputs_dir)
        m = match(r"^RYF_sxtdeg_checkpoint_rank0_iteration(\d+)\.jld2$", name)
        m === nothing && continue
        push!(checkpoint_iters, parse(Int, m.captures[1]))
    end

    sort!(unique!(checkpoint_iters))
    return checkpoint_iters
end

seen_iterations = Set{Int}()
time_days = Float64[]
iterations = Int[]

umax = Float64[]
vmax = Float64[]
wmax = Float64[]

extrema_max = Dict{String, Vector{Float64}}()
extrema_min = Dict{String, Vector{Float64}}()

for logfile in logfiles
    @info "Parsing $logfile"

    for line in eachline(logfile)
        occursin("┌ Info:", line) || continue
        occursin("iteration:", line) || continue
        occursin("extrema(", line) || continue

        tm = match(r"time:\s*([-\d\.eE+NaInf]+)\s*(seconds?|hours?|days?),\s*iteration:\s*(\d+)", line)
        tm === nothing && continue

        t = parse_time_days(tm.captures[1], lowercase(tm.captures[2]))
        it = parse(Int, tm.captures[3])
        it in seen_iterations && continue
        push!(seen_iterations, it)

        push!(time_days, t)
        push!(iterations, it)

        for values in values(extrema_max)
            push!(values, NaN)
        end
        for values in values(extrema_min)
            push!(values, NaN)
        end

        um, vm, wm = NaN, NaN, NaN
        if (m = match(r"max\|u\|:\s*\(\s*([-\d\.eE+NaInf]+),\s*([-\d\.eE+NaInf]+),\s*([-\d\.eE+NaInf]+)\s*\)", line)) !== nothing
            um = parse_float_maybe(m.captures[1])
            vm = parse_float_maybe(m.captures[2])
            wm = parse_float_maybe(m.captures[3])
        end
        push!(umax, um)
        push!(vmax, vm)
        push!(wmax, wm)

        for m in eachmatch(r"extrema\(([^)]+)\):\s*\(\s*([-\d\.eE+NaInf]+),\s*([-\d\.eE+NaInf]+)\s*\)", line)
            var = strip(m.captures[1])
            v_max = parse_float_maybe(m.captures[2])
            v_min = parse_float_maybe(m.captures[3])

            if !haskey(extrema_max, var)
                extrema_max[var] = fill(NaN, length(time_days))
                extrema_min[var] = fill(NaN, length(time_days))
            end

            extrema_max[var][end] = v_max
            extrema_min[var][end] = v_min
        end
    end
end

isempty(time_days) && error("No extrema timesteps were parsed from the provided stderr logs.")

perm = sortperm(iterations)
time_days = time_days[perm]
iterations = iterations[perm]
umax = umax[perm]
vmax = vmax[perm]
wmax = wmax[perm]

for (name, values) in extrema_max
    extrema_max[name] = values[perm]
    extrema_min[name] = extrema_min[name][perm]
end

checkpoint_iterations = checkpoint_iterations_from_outputs(outputs_dir)
checkpoint_times_days = Float64[]
for cpi in checkpoint_iterations
    t = checkpoint_iteration_to_time_days(cpi, iterations, time_days)
    isfinite(t) && push!(checkpoint_times_days, t)
end

preferred_order = ["T", "S", "η"]
all_extrema_terms = collect(keys(extrema_max))

ordered_extrema_terms = String[]
for term in preferred_order
    term in all_extrema_terms && push!(ordered_extrema_terms, term)
end
for term in sort(all_extrema_terms)
    term in preferred_order && continue
    push!(ordered_extrema_terms, term)
end

units = Dict(
    "T" => "°C",
    "S" => "g/kg",
    "η" => "m",
    "|u|" => "m s⁻¹",
    "|v|" => "m s⁻¹",
    "|w|" => "m s⁻¹",
)

num_panels = length(ordered_extrema_terms) + 3
ncols = 2
nrows = cld(num_panels, ncols)
fig = Figure(size = (1400, max(350 * nrows, 900)))

abs_panels = [
    ("|u|", umax),
    ("|v|", vmax),
    ("|w|", wmax),
]

let panel_index = 1
    for term in ordered_extrema_terms
        row = cld(panel_index, ncols)
        col = mod1(panel_index, ncols)
        ylabel = get(units, term, "value")

        ax = Axis(
            fig[row, col],
            title = "extrema($term)",
            xlabel = "Time (days)",
            ylabel = ylabel,
        )

        lines!(ax, time_days, extrema_max[term], color = :firebrick, label = "max")
        lines!(ax, time_days, extrema_min[term], color = :royalblue, label = "min")

        if !isempty(checkpoint_times_days)
            vlines!(ax, checkpoint_times_days, color = :black, linestyle = :dash, linewidth = 1.5)
        end

        axislegend(ax, position = :lt)
        panel_index += 1
    end

    for (term, values) in abs_panels
        row = cld(panel_index, ncols)
        col = mod1(panel_index, ncols)
        ylabel = get(units, term, "value")

        ax = Axis(
            fig[row, col],
            title = "max$term",
            xlabel = "Time (days)",
            ylabel = ylabel,
        )

        lines!(ax, time_days, values, color = :darkgreen, label = "max")

        if !isempty(checkpoint_times_days)
            vlines!(ax, checkpoint_times_days, color = :black, linestyle = :dash, linewidth = 1.5)
        end

        axislegend(ax, position = :lt)
        panel_index += 1
    end
end

save(output_figure, fig)

@info "Saved figure to $output_figure"
@info "Parsed $(length(time_days)) unique timesteps across $(length(logfiles)) logs."
@info "Checkpoint iterations from outputs: $(checkpoint_iterations)"
@info "Checkpoint marker times (days): $(checkpoint_times_days)"
