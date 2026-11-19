# Global ocean heat content ρ₀cₚ∫Θ dV and freshwater content ∫(S₀ − S)/S₀ dV (S₀ = 35) from the 5-day global integrals.
# `include` in the persistent REPL.
using CairoMakie, JLD2

let pp = "/data/gpfs/projects/punim2499/taimoor/gadi-ensembles/postprocessed/RYF_control/timeseries",
    figdir = "/data/gpfs/projects/punim2499/taimoor/gadi-ensembles/figures/RYF_control/compare",
    ρc = 1026 * 3991.87, S₀ = 35.0, yr = 365 * 86400.0
    t = Float64[]; Ti = Float64[]; Si = Float64[]; V = Float64[]
    for f in readdir(pp, join=true)
        b = basename(f)
        (startswith(b, "global_integrals") || startswith(b, "integrals_5day")) || continue
        d = load(f); append!(t, d["t"]); append!(Ti, d["T_integral"]); append!(Si, d["S_integral"]); append!(V, d["volume"])
    end
    p = sortperm(t); t, Ti, Si, V = t[p], Ti[p], Si[p], V[p]
    keep = [true; diff(t) .> 0]; t, Ti, Si, V = t[keep], Ti[keep], Si[keep], V[keep]   # restarts can repeat records

    for (name, title, ylabel, y) in (("ohc", "Ocean Heat Content", "OHC (J)", ρc .* Ti),
                                     ("freshwater_content", "Ocean Freshwater Content", "FW (m³)", (S₀ .* V .- Si) ./ S₀))
        fig = Figure(size=(1000, 500), fontsize=16)
        ax = Axis(fig[1, 1]; title, xlabel="Time (years)", ylabel)
        lines!(ax, t ./ yr, y; color=:steelblue, linewidth=1.5)
        xlims!(ax, extrema(t ./ yr)...)
        save(joinpath(figdir, name * ".png"), fig; px_per_unit=2)
        println("SAVED ", joinpath(figdir, name * ".png"))
    end
    println("record ", round(t[1] / yr, digits=2), "–", round(t[end] / yr, digits=2), " years")
end
