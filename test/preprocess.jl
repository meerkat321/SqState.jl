@testset "preprocess for training sqth_th" begin
    n = 6
    file = joinpath(SqState.training_data_path(), "test.jld2")

    points, 𝛒s, args, σs = gen_data_sqth_th(n_data=n)
    jldsave(file; points, 𝛒s, args, σs)

    loader = SqState.preprocess_q2σs(file, batch_size=2)
    x, y = first(loader)
    @test size(x) == (2, 4096, 2)
    @test size(y) == (4096, 2)

    loader = SqState.preprocess_q2ρ(file, batch_size=2)
    x, y = first(loader)
    @test size(x) == (2, 4096, 2)
    @test size(y) == (2, 100, 100, 2)

    loader = SqState.preprocess_q2args(file, batch_size=2)
    x, y = first(loader)
    @test size(x) == (2, 4096, 2)
    @test size(y) == (6, 2)

    rm(file)
end

@testset "preprocess for training sqth" begin
    n = 6
    file = joinpath(SqState.training_data_path(), "test.jld2")

    points, 𝛒s, args, σs = gen_data_sqth(n_data=n)
    jldsave(file; points, 𝛒s, args, σs)

    loader = SqState.preprocess_q2σs(file, batch_size=2)
    x, y = first(loader)
    @test size(x) == (2, 4096, 2)
    @test size(y) == (4096, 2)

    loader = SqState.preprocess_q2ρ(file, batch_size=2)
    x, y = first(loader)
    @test size(x) == (2, 4096, 2)
    @test size(y) == (2, 100, 100, 2)

    loader = SqState.preprocess_q2args(file, batch_size=2)
    x, y = first(loader)
    @test size(x) == (2, 4096, 2)
    @test size(y) == (3, 2)

    rm(file)
end
