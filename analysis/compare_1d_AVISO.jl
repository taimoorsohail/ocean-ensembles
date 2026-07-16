# Compare area-weighted monthly global mean sea level (GMSL) from the model
# with the AVISO monthly ADT climatology. The 2D comparison implementation is
# reused so both analyses have identical temporal bins, spatial interpolation,
# wet masking, and climatology-month alignment.

if !isdefined(@__MODULE__, :build_aviso_climatology_cache)
    include(joinpath(@__DIR__, "compare_2d_AVISO.jl"))
end

const AVISO_GMSL_FIGURE = get(ENV, "COMPARE_1D_AVISO_FIGURE",
                              joinpath(FIGDIR, "compare_1d_AVISO_GMSL_$(RESOLUTION).png"))

function comparison_cell_areas(target, model_grid)
    grid = target == :model ? underlying_grid(model_grid) : aviso_source_template().grid
    hasproperty(grid, :Azᶜᶜᵃ) || error("Comparison grid does not contain center-cell areas.")
    return physical_matrix(getproperty(grid, :Azᶜᶜᵃ), grid; T = Float64)
end

function common_area_weighted_gmsl(model, aviso, areas, mask)
    size(model) == size(aviso) == size(areas) || error("GMSL field and area shapes do not align.")
    !isnothing(mask) && size(mask) != size(model) && error("GMSL mask shape does not align with the fields.")

    model_integral = 0.0
    aviso_integral = 0.0
    wet_area = 0.0
    @inbounds for i in eachindex(model, aviso, areas)
        (isnothing(mask) || mask[i]) || continue
        model_value = Float64(model[i])
        aviso_value = Float64(aviso[i])
        area = Float64(areas[i])
        if isfinite(model_value) && isfinite(aviso_value) && isfinite(area) && area > 0
            model_integral += model_value * area
            aviso_integral += aviso_value * area
            wet_area += area
        end
    end

    wet_area > 0 || error("No common finite ocean area was found for the GMSL comparison.")
    return model_integral / wet_area, aviso_integral / wet_area
end

function build_AVISO_GMSL_series(metadata, aviso_cache, areas, mask)
    nframes = length(metadata.bins)
    aviso_months = mod1.(metadata.bins, 12)
    model = load_aviso_cached_frame(metadata.cache_path, SSH_VARIABLE, 1)
    aviso = load_aviso_cached_frame(aviso_cache, SSH_VARIABLE, aviso_months[1])
    model_reader = aviso_cached_frame_reader(metadata.cache_path)
    aviso_reader = aviso_cached_frame_reader(aviso_cache)
    model_gmsl = Vector{Float64}(undef, nframes)
    aviso_gmsl = Vector{Float64}(undef, nframes)

    try
        for frame in 1:nframes
            month = aviso_months[frame]
            model_reader.load_frame!(model, SSH_VARIABLE, frame) || error("Could not load model month $(frame).")
            aviso_reader.load_frame!(aviso, SSH_VARIABLE, month) || error("Could not load AVISO month $(month).")
            model_gmsl[frame], aviso_gmsl[frame] = common_area_weighted_gmsl(model, aviso, areas, mask)
        end
    finally
        model_reader.close_reader!()
        aviso_reader.close_reader!()
    end

    model_gmsl .-= first(model_gmsl)
    aviso_gmsl .-= first(aviso_gmsl)
    time_years = Float64.(metadata.midpoints) ./ SECONDS_PER_YEAR
    return (; time_years, model = model_gmsl, aviso = aviso_gmsl,
            difference = model_gmsl .- aviso_gmsl,
            bins = metadata.bins, months = aviso_months)
end

function save_AVISO_GMSL_figure(series; output_path = AVISO_GMSL_FIGURE)
    fig = Figure(size = (1200, 850))
    ax_gmsl = Axis(fig[1, 1],
                   title = "Monthly global mean sea-level change from first month",
                   ylabel = "GMSL change (m)")
    ax_difference = Axis(fig[2, 1],
                         title = "Model - AVISO",
                         xlabel = "RYF model year",
                         ylabel = "GMSL-change difference (m)")

    lines!(ax_gmsl, series.time_years, series.model; color = :dodgerblue4, linewidth = 2, label = "Model")
    lines!(ax_gmsl, series.time_years, series.aviso; color = :black, linewidth = 2, label = "AVISO climatology")
    lines!(ax_difference, series.time_years, series.difference; color = :firebrick, linewidth = 2)
    hlines!(ax_difference, 0; color = (:black, 0.45), linestyle = :dash)
    axislegend(ax_gmsl, position = :rb)
    linkxaxes!(ax_gmsl, ax_difference)
    hidexdecorations!(ax_gmsl; grid = false)

    mkpath(dirname(output_path))
    save(output_path, fig; px_per_unit = 2)
    @info "Saved AVISO GMSL comparison figure." output_path months = length(series.bins)
    return output_path
end

function run_AVISO_GMSL_comparison()
    mkpath(FIGDIR)
    mkpath(ANALYSIS_OUTPUT_PATH)
    scratch_dir = mktempdir(ANALYSIS_OUTPUT_PATH; prefix = "compare_1d_AVISO_scratch_")
    try
        index = index_ssh_intervals()
        target = comparison_target(index.grid)
        aviso_cache = build_aviso_climatology_cache(index.grid, target)
        model_cache = build_monthly_model_cache(index, target, joinpath(scratch_dir, "monthly_model_SSH.jld2"))
        mask = comparison_mask(target, index.grid, aviso_cache)
        areas = comparison_cell_areas(target, index.grid)
        series = build_AVISO_GMSL_series(model_cache, aviso_cache, areas, mask)
        return save_AVISO_GMSL_figure(series)
    finally
        rm(scratch_dir; recursive = true, force = true)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_AVISO_GMSL_comparison()
end
