using DataDeps
using QuantumStateBase

@testset "gen_data" begin
    n = 2

    points, 𝛒s, args, σs = gen_data_sqth_th(n_data=n)
    @test all([
        size(points) == (2, 4096, n),
        size(𝛒s) == (100, 100, n),
        size(args) == (6, n)
    ])

    points, 𝛒s, args, σs = gen_data_sqth(n_data=n)
    @test all([
        size(points) == (2, 4096, n),
        size(𝛒s) == (100, 100, n),
        size(args) == (3, n)
    ])
end
