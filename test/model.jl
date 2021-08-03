@testset "model" begin
    m = model()

    loader = preprocess("ci.jld2", batch_size=1, fragment_size=2)
    x, y = first(loader)

    ŷ = m(x)
    @test size(ŷ) == (3, 1)
end
