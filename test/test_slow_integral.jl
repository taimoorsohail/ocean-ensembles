using Oceananigans
using Statistics: mean
using OceanEnsembles
using BenchmarkTools
using Oceananigans.Operators: Δx, Δy, Δz
using ClimaOcean

arch = CPU()
grid = TripolarGrid(arch;
                    size = (10,10,10),
                    z = (-6000,0),
                    halo = (6, 6, 3),
                    first_pole_longitude = 70,
                    north_poles_latitude = 55)

free_surface = SplitExplicitFreeSurface(grid)

ocean = ocean_simulation(grid; free_surface)
u = XFaceField(grid);

set!(u, 1)

array_mask = ones(Float64, size(u))  # shape: (Nx+1, Ny, Nz) for XFaceField
array_mask[1:5, :, :] .= 0  # Example: zero out part of domain
bit_mask = BitArray(array_mask .!= 0.0)
field_mask = XFaceField(grid)
set!(field_mask, (x, y, z) -> if x > 0.5 1.0 else 0.0 end)

@btime u_bit_masked = compute!(Integral(u, dims = (1,2,3), condition=bit_mask))
@btime u_bit_masked = compute!(Integral(u, dims = (1,2,3), condition=array_mask))
@btime u_field_masked = compute!(Integral(u, dims = (1,2,3), condition=field_mask))
@btime u_bit_masked = compute!(Integral(u, dims = (1,2,3)))
@btime u_bit_summed = sum(u*Δx*Δy*Δz * bit_mask, dims = (1,2,3))
# @btime u_bit_summed = sum(u*Ax*Ay*Az .* bit_mask, .dims = (1,2,3))

u_bit_masked = compute!(Integral(u, dims = (1,2,3), condition=bit_mask))
u_bit_summed = sum(u*Δx*Δy*Δz * bit_mask, dims = (1,2,3))


# @btime u_field_summed = sum(interior(u * field_mask), dims = (1,2,3))

u_field_masked = compute!(Integral(u, dims = (1,2,3), condition=field_mask))
u_field_summed = sum(interior(u * field_mask), dims = (1,2,3))

# @show u_bit_masked[1,1,1]
# @show u_bit_summed[1,1,1]

# @show u_field_masked[1,1,1]
# @show u_field_summed[1,1,1]