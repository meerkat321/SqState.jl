using SqState
using Test

ENV["GKSwstype"]="nul"

@testset "SqState.jl" begin

    include("state/state.jl")
    include("data_generator.jl")

    include("read.jl")
    include("polynomial.jl")
    include("wigner.jl")
    include("plot.jl")

end
