module Diagnostics

    using Oceananigans
    using Oceananigans.Fields: location, ReducedField

    export ocean_tracer_content!, volume_transport!

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

    function volume_transport!(names, ∫outputs; outputs, operators, dims, condition, suffix::AbstractString)
        if length(outputs) == length(operators)
            for (i, key) in enumerate(keys(outputs))
                if key == :w
                    cond3d = condition[2]
                else
                    cond3d = condition[1]
                end

                f = outputs[key]
                
                ∫f = sum(f * operators[i]; dims, condition = cond3d)
                push!(∫outputs, ∫f)
                push!(names, Symbol(key, suffix))

                LX, LY, LZ = location(f)
                onefield = Field{LX, LY, LZ}(f.grid)
                set!(onefield, 1)
                
                ∫dV = sum(onefield * operators[i]; dims, condition = cond3d)
                push!(∫outputs, ∫dV)
                push!(names, Symbol(:dV, "_$(key)", suffix))
            end
        else
            throw(ArgumentError("The number of operators must be equal to the number of tracers"))
        end
        return names, ∫outputs
    end
end
