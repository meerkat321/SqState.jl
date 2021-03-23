using SqState
using Test

@testset "SqState.jl" begin

    include("data_generator/data_generator.jl")

    include("read.jl")
    include("utils.jl")
    include("polynomial.jl")
    include("wigner.jl")
    include("plot.jl")

end
