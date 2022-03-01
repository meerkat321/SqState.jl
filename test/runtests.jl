using SqState
using LinearAlgebra
using Flux
using JLD2
using Test

@testset "SqState.jl" begin
    #include("utils.jl")
    include("gen_data.jl")
    #include("preprocess.jl")
    #include("model.jl")
end
