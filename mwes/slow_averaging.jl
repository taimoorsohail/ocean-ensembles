using Oceananigans
using JLD2
using Glob


output_path = expanduser("/g/data/v46/txs156/ocean-ensembles/outputs/saved/")

# Define variables in the files
vars = [ "T",
 "S",
 "u",
 "v",
 "w"]

 # Define function to create dictionary of FieldTimeSeries objects
function create_dict(vars, path)
    dicts = Dict()
    for var in vars
        try
            # Surface
            @info var
            dicts[var] = FieldTimeSeries(path, var)
        catch e
            if e isa KeyError
                @warn "Skipping variable $var: Key not found in file."
            else
                rethrow(e)
            end
        end
    end
    return dicts
end

# Find the matching files for depth 3m and iteration 0
pattern = "global_3m_*onedeg_RYF_iteration0.jld2"
matching_files = glob(pattern, output_path)
# Create a dictionary of FieldTimeSeries objects
slice = create_dict(vars, matching_files[1])

# Loop over vars and times to compute averages
for var in vars
    for (time_index, time) in enumerate(slice[var].times)
        field = slice[var][time_index]
        avg_field = Average(field)
        @time test = Field(avg_field)[1,1,1]
    end
end
