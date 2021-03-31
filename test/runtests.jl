using SqState
using Test

@testset "SqState.jl" begin

    include("state.jl")

    include("read.jl")
    include("utils.jl")
    include("polynomial.jl")
    include("wigner.jl")
    include("plot.jl")

end
