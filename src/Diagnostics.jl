module Diagnostics

    using Oceananigans
    using Oceananigans.Fields: location

    export integrate_tracer, integrate_transport

    function integrate_tracer(outputs; metric, dims, condition, suffix::AbstractString)
        names = []
        ∫outputs = []
        for key in keys(outputs)
            f = outputs[key]
            ∫f = Integral(f * metric; dims, condition)
            push!(∫outputs, ∫f)
            push!(names, Symbol(key, suffix))
        end

        onefield = CenterField(first(outputs).grid)
        set!(onefield, 1)

        ∫dV = Integral(onefield * metric; dims, condition)
        push!(∫outputs, ∫dV)
        push!(names, Symbol(:dV, suffix))

        return NamedTuple{Tuple(names)}(∫outputs)
    end

    function integrate_transport(outputs; metrics, dims, condition, suffix::AbstractString)
        names = []
        ∫outputs = []
        for (i, key) in enumerate(keys(outputs))
            f = outputs[key]
            ∫f = sum(f * metrics[i]; dims, condition)
            push!(∫outputs, ∫f)
            push!(names, Symbol(key, suffix))

            LX, LY, LZ = location(f)
            onefield = Field{LX, LY, LZ}(f.grid)
            set!(onefield, 1)

            ∫dV = sum(onefield * metrics[i]; dims, condition)
            push!(∫outputs, ∫dV)
            push!(names, Symbol(:dV, "_$(key)", suffix))
        end

        return NamedTuple{Tuple(names)}(∫outputs)
    end
end