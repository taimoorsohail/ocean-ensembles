module OceanEnsembles

export basin_mask, integrate_tracer, integrate_transport

include("BasinMask.jl")
include("Diagnostics.jl")

using .BasinMask
using .Diagnostics

end # module OceanEnsembles
