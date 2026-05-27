using Oceananigans
using Printf

"""
Minimal reproducer for a 2D `(Center, Center, Nothing)` field that comes from a
Field-wrapped computation and is written with `JLD2Writer`.

This tests three cases on an immersed-boundary grid:

1. A plain in-memory 2D field.
2. An in-memory 2D field created via `Field(Integral(..., dims=(3)))`.
3. The same computed 2D field after `JLD2Writer` save and reload.

Case 3 is the closest match so far to the failing analysis path, where a
Field-wrapped computation is saved by an output writer and later reloaded via
`FieldTimeSeries(path, name)`.
"""

Nx = 4
Ny = 4
Nz = 4
outdir = mktempdir()
outpath = joinpath(outdir, "wrapped_output.jld2")

simple_grid = RectilinearGrid(size = (Nx, Ny, Nz), extent = (1, 1, 1))
bottom_height = Field{Center, Center, Nothing}(simple_grid)
set!(bottom_height, -0.8)
immersed_grid = ImmersedBoundaryGrid(simple_grid, GridFittedBottom(bottom_height))

free_surface = SplitExplicitFreeSurface(immersed_grid; substeps = 1)
model = HydrostaticFreeSurfaceModel(; grid = immersed_grid,
                                      free_surface,
                                      tracers = (:c,))
set!(model.tracers.c, 1)

plain_2d = Field{Center, Center, Nothing}(immersed_grid)
set!(plain_2d, 1)
reduced_plain = Integral(plain_2d, dims = (1, 2)) |> Field
compute!(reduced_plain)
value_plain = interior(reduced_plain)[1, 1, 1]
@printf("Plain 2D field horizontal integral succeeded: %.8f\n", value_plain)

wrapped_2d = Field(Integral(model.tracers.c, dims = (3)))
compute!(wrapped_2d)
reduced_wrapped = Integral(wrapped_2d, dims = (1, 2)) |> Field
compute!(reduced_wrapped)
value_wrapped = interior(reduced_wrapped)[1, 1, 1]
@printf("In-memory wrapped-computation field horizontal integral succeeded: %.8f\n", value_wrapped)

simulation = Simulation(model; Δt = 0.01, stop_iteration = 1)
outputs = (; wrapped_2d = Field(Integral(model.tracers.c, dims = (3))))
simulation.output_writers[:wrapped] = JLD2Writer(model, outputs;
                                                 dir = outdir,
                                                 filename = "wrapped_output",
                                                 schedule = IterationInterval(1),
                                                 overwrite_existing = true)
run!(simulation)

loaded_fts = FieldTimeSeries(outpath, "wrapped_2d")
loaded_snapshot = loaded_fts[1]

@show typeof(loaded_snapshot)
@show location(loaded_snapshot)
@show size(loaded_snapshot)

reduced_loaded = Integral(loaded_snapshot, dims = (1, 2)) |> Field
compute!(reduced_loaded)
value_loaded = interior(reduced_loaded)[1, 1, 1]
@printf("Reloaded wrapped-computation field horizontal integral succeeded: %.8f\n", value_loaded)

println("Temporary files written to: $outdir")
