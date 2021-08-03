using SqState
using Test

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

@testset "SqState.jl" begin
    @time SqState.model_path()
end
