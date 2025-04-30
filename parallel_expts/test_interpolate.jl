using Oceananigans
using Oceananigans.Fields: interpolate!
using MPI

# Automatically distributes among available processors
arch = Distributed(GPU())

rank = arch.local_rank
Nranks = MPI.Comm_size(arch.communicator)
println("Hello from process $rank out of $Nranks")

x = y = z = (0, 1)
grid = RectilinearGrid(arch; size=(64, 64, 64), x, y, z, topology=(Periodic, Periodic, Bounded))

@info "The grid on rank $rank:"
@info "$grid"

c = CenterField(grid)
set!(c, (x, y, z) -> x * y^2 * z^3)

@info "c on rank $rank:"
@show c

u = XFaceField(grid)
set!(c, (x, y, z) -> x * y^2 * z^3)
interpolate!(u, c)

@info "u on rank $rank:"
@show u
