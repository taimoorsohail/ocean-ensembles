using Oceananigans
using BenchmarkTools

Nx, Ny, Nz = 360, 180, 40

underlying_grid = TripolarGrid(size = (Nx, Ny, Nz), z=(-4000, 0))

h₀ = 6000
width = 60 # degrees
hill(λ, φ) = h₀ * exp(-(λ^2 + φ^2) / 2width^2)
bottom(λ, φ) = - underlying_grid.Lz + hill(λ, φ)

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

c = Field{Center, Center, Nothing}(grid)
set!(c, rand())

area = Field{Center, Center, Nothing}(grid)
set!(area, 1)
area = (Integral(area, dims = (1,2)) |> Field)[1,1,1]

@benchmark c̄_avg = (Average(c, dims = (1,2)) |> Field)[1,1,1]
@benchmark c̄_int = (Integral(c, dims = (1,2)) |> Field)[1,1,1] / area

c̄_avg = (Average(c, dims = (1,2)) |> Field)[1,1,1]
c̄_int = (Integral(c, dims = (1,2)) |> Field)[1,1,1] / area

@assert c̄_avg ≈ c̄_int