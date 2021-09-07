@testset "preprocess for training" begin
    n = 6
    points, 𝛒s, args = gen_data(n_data=n)
    jldsave(joinpath(SqState.training_data_path(), "test.jld2"); points, 𝛒s, args)

    loader = SqState.preprocess_q2args("test.jld2", batch_size=2)
    x, y = first(loader)

    @test size(x) == (4096, 1, 2)
    @test size(y) == (6, 2)

    rm(joinpath(SqState.training_data_path(), "test.jld2"))
end
