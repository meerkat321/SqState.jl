@testset "gen data" begin
    @test 5 ≤ SqState.rand2range((5, 10)) < 10

    r_range = (0, 2)
    θ_range = (0, 2π)
    n̄_range = (0, 0.5)

    args = SqState.rand_arg_sqth_th(r_range, θ_range, n̄_range)
    @test all([
        r_range[1] ≤ args[1] < r_range[2],
        θ_range[1] ≤ args[2] < θ_range[2],
        n̄_range[1] ≤ args[3] < n̄_range[2],
        n̄_range[1] ≤ args[4] < n̄_range[2],
        0 ≤ args[5] < 1,
        args[6] == 1-args[5]
    ])

    args = SqState.rand_arg_sqth(r_range, θ_range, n̄_range)
    @test all([
        r_range[1] ≤ args[1] < r_range[2],
        θ_range[1] ≤ args[2] < θ_range[2],
        n̄_range[1] ≤ args[3] < n̄_range[2],
    ])

    @test tr(SqState.construct_state_sqth_th(SqState.rand_arg_sqth_th(r_range, θ_range, n̄_range)..., 100).𝛒) ≈ 1

    @test tr(SqState.construct_state_sqth(SqState.rand_arg_sqth(r_range, θ_range, n̄_range)..., 100).𝛒) ≈ 1
end

@testset "model" begin
    a = rand(5, 5, 10)
    b = rand(5, 5, 10)
    c = SqState.to_complex(a, b)
    @test size(c) == (5, 5, 10)
    @test all([c[i] == complex(a[i], b[i]) for i in 1:length(a)])

    @test size(SqState.reshape_cholesky(rand(5*5, 10))) == (5, 5, 10)
    @test size(SqState.cholesky2ρ(rand(5*5, 10))) == (2, 5, 5, 10)

    @test size(SqState.res_block((8, 4, 4, 1), (1, 15, 7), (0, 7, 3), 1, 0, 2)(rand(Float32, 1024, 8, 2))) == (512, 1, 2)
end

@testset "training" begin
    file = joinpath(SqState.model_path(), "test.jld2")
    SqState.update_model!(SqState.model_path(), "test", 1)
    @test isfile(file)
    rm(file)

    device = SqState.get_device()
    @test device == Flux.cpu || device == Flux.gpu
end

@testset "inference" begin
    m = SqState.get_model("q2args_sqth")
    @test size(m(rand(Float32, 2, 4096, 1))) == (3, 1)
end
