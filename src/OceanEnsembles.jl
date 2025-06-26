module OceanEnsembles

export basin_mask, ocean_tracer_content!, volume_transport!, combine_outputs, identify_combination_targets

include("BasinMask.jl")
include("Diagnostics.jl")
include("OutputWrangling.jl")

using .BasinMask
using .Diagnostics
using .OutputWrangling

end # module OceanEnsembles
