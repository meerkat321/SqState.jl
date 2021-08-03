@testset "preprocess for training" begin
    loader = preprocess("ci.jld2", batch_size=1, fragment_size=2)
    x, y = first(loader)

    @test size(x) == (4096, 1, 1)
    @test size(y) == (3, 1)
end
