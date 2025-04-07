module Diagnostics

using Oceananigans
export integrate_tuple

function integrate_tuple(outputs; volmask, dims, condition, suffix::AbstractString)
    names = []
    ∫outputs = []
    for key in keys(outputs)
        f = outputs[key]
        ∫f = Integral(f; dims, condition)
        push!(∫outputs, ∫f)
        push!(names, Symbol(key, suffix))
    end

    ∫dV = Integral(volmask; dims, condition)
    push!(∫outputs, ∫dV)
    push!(names, Symbol(:dV, suffix))

    return NamedTuple{Tuple(names)}(∫outputs)
end

