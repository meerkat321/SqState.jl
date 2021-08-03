using SqState
using Test

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

@testset "SqState.jl" begin
    include("gen_data.jl")
    include("preprocess.jl")
    include("model.jl")

    rm(joinpath(SqState.training_data_path(), "ci.jld2"))
end
