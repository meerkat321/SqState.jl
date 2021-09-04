@testset "model" begin
    m = model()

    n = 6
    gen_data(n_data=n, file_name="ci")
    loader = preprocess("ci.jld2", batch_size=2)
    x, y = first(loader)

    ŷ = m(x)
    @test size(ŷ) == (6, 2)

    rm(joinpath(SqState.training_data_path(), "ci.jld2"))
    rm(joinpath(SqState.training_data_path(), "../mat_data/ci.mat"))
end
