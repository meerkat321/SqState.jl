@testset "gen data" begin
    @test 99 ≤ SqState.rand2range((99, 101)) < 101

    r, θ, n̄, c1, c2, c3 = SqState.rand_arg((0, 3), (0, 2π), (0, 2))
    @test all([
        0 ≤ r < 3,
        0 ≤ θ < 2π,
        0 ≤ n̄ < 2,
        0 ≤ c1 < 1,
        0 ≤ c2 < 1,
        0 ≤ c3 < 1,
        c1 + c2 + c3 ≈ 1
    ])

    @test tr(SqState.construct_state(r, θ, n̄, c1, c2, c3, 200).𝛒) ≈ 1
end

@testset "inference" begin
    # TODO: inference test
end
