@testset "model" begin
    m = model_q2args_sqth()

    n = 6
    points, 𝛒s, args, σs = gen_data_sqth(n_data=n)
    jldsave(joinpath(SqState.training_data_path(), "test.jld2"); points, 𝛒s, args, σs)

    loader = SqState.preprocess_q2args(joinpath(SqState.training_data_path(), "test.jld2"), batch_size=2)
    x, y = first(loader)

    ŷ = m(x)
    @test size(ŷ) == (3, 2)

    rm(joinpath(SqState.training_data_path(), "test.jld2"))
end
