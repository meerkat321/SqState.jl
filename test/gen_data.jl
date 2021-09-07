using DataDeps
using QuantumStateBase

@testset "gen_data" begin
    n = 2

    points, 𝛒s, args = gen_data(n_data=n)
    @test all([
        size(points) == (2, 4096, n),
        size(𝛒s) == (n, ),
        size(args) == (6, n)
    ])
end
