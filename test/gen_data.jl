using DataDeps
using QuantumStateBase
using KernelDensity

@testset "utils" begin
    # download data set
    @time @test SqState.data_path() == joinpath(datadep"SqState", "data")

    n = 1e5

    trapezoid_area(top, bottom, n) = (top + bottom) * n / 2

    int_result = sum(SqState.rand2range((5, 10)) for _ in 1:n)
    @test isapprox(int_result, trapezoid_area(5, 10, n), rtol=1e-3)

    r, θ, n̄, c1, c2, c3 = SqState.rand_arg((1, 2), (3, 4), (5, 6))
    @test all([1≤r<2, 3≤θ<4, 5≤n̄<6, 0≤c1<1, 0≤c2<1, 0≤c3<1])
    @test c1+c2+c3 ≈ 1
end

@testset "gen_data" begin
    n = 2

    points, 𝛒s, args = gen_data(n_data=n, file_name="ci")
    @test "ci.jld2" in readdir(SqState.training_data_path())
    @test "ci.mat" in readdir(SqState.training_data_path())

    θs = LinRange(0, 2π, 10)
    xs = LinRange(-10, 10, 10)
    point_dim = 900
    n_points = 4096
    for i in 1:n
        r, θ, n̄, c1, c2, c3 = args[:, i]
        state =
            c1 * SqueezedState(ξ(r, θ), dim=point_dim, rep=StateMatrix) +
            c2 * SqueezedThermalState(ξ(r, θ), n̄, dim=point_dim) +
            c3 * ThermalState(n̄, dim=point_dim)

        @test 𝛒s[i] == 𝛒(state)
    end

    rm(joinpath(SqState.training_data_path(), "ci.jld2"))
    rm(joinpath(SqState.training_data_path(), "ci.mat"))
end

@testset "gen_non_gaussian_data" begin
    gen_non_gaussian_data()
end
