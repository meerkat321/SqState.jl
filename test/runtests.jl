using SqState
using Test

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

@testset "SqState.jl" begin
    include("gen_data.jl")
    include("preprocess.jl")
end
