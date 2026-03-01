include("analysis_2d_horizontal_forcing.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    make_forcing_animation()
end
