module Diagnostics

    using Oceananigans
    using Oceananigans.Fields: location

    export ocean_tracer_content!, volume_transport

    function ocean_tracer_content!(names, ∫outputs; outputs, operator, dims, condition, suffix::AbstractString)
        for key in keys(outputs)
            f = outputs[key]
            ∫f = Integral(f * operator; dims, condition)
            push!(∫outputs, ∫f)
            push!(names, Symbol(key, suffix))
        end

        onefield = CenterField(first(outputs).grid)
        set!(onefield, 1)

        ∫dV = Integral(onefield * operator; dims, condition)
        push!(∫outputs, ∫dV)
        push!(names, Symbol(:dV, suffix))

        return names, ∫outputs
    end

    function volume_transport(outputs; operators, dims, condition, suffix::AbstractString)
        names = Symbol[]
        ∫outputs = Reduction[]
        if length(outputs) == length(operators)
            for (i, key) in enumerate(keys(outputs))
                f = outputs[key]
                ∫f = sum(f * operators[i]; dims, condition)
                push!(∫outputs, ∫f)
                push!(names, Symbol(key, suffix))

                LX, LY, LZ = location(f)
                onefield = Field{LX, LY, LZ}(f.grid)
                set!(onefield, 1)

                ∫dV = sum(onefield * operators[i]; dims, condition)
                push!(∫outputs, ∫dV)
                push!(names, Symbol(:dV, "_$(key)", suffix))
            end
        else
            throw(ArgumentError("The number of operators must be equal to the number of tracers"))
        end
        return NamedTuple{Tuple(names)}(∫outputs)
    end
end
