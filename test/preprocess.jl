@testset "preprocess" begin
    l = [1, 2, 3, 4, 5]
    𝐜 = l * l'

    @test isapprox(
        SqState.𝛒2y(𝐜, δ=1e-10),
        [
            5,
            4, 4.5e-5,
            3, 3.6e-5, 1.8e-5,
            2, 2.7e-5, 1.4e-5, 9.8e-6,
            1, 2.2e-5, 1.7e-5, 1.5e-5, 1.4e-5,
            0,
            0, 0,
            0, 0, 0,
            0, 0, 0, 0
        ],
        atol=1e-5
    )

    file_names = readdir(SqState.training_data_path())
    loader = preprocess(file_names[1], batch_size=10)
    x, y = first(loader)
    @test size(x) == (4096, 1, 10)
    @test size(y) == (70*70, 10)
end
