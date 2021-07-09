using SqState
using Test

ENV["GKSwstype"]="nul"

@testset "SqState.jl" begin
    # include("plot.jl")
    # include("model.jl")
    include("preprocess.jl")
end
