using SqState
using Test

@testset "SqState.jl" begin

    include("state.jl")

    include("read.jl")
    include("polynomial.jl")
    include("utils.jl")
    include("wigner.jl")
    include("plot.jl")

end
