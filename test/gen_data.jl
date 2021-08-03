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

    r, θ, n̄, θ_offset = SqState.rand_arg((1, 2), (3, 4), (5, 6), (7, 8))
    @test all([1≤r<2, 3≤θ<4, 5≤n̄<6, 7≤θ_offset<8])
end

@testset "gen_squeezed_thermal_data" begin
    n = 2

    points, 𝛒s, args,
    n_data, n_points,
    r_range, θ_range, n̄_range, θ_offset_range,
    point_dim, label_dim = gen_squeezed_thermal_data(
        n_data=n, n_points=4096,
        r_range=(0, 2), θ_range=(0, 2π), n̄_range=(0, 0.5), θ_offset_range=(0, 0),
        point_dim=500, label_dim=70,
        file_name="ci"
    )

    @test all([
        n_data==n, n_points==4096,
        r_range==(0, 2), θ_range==(0, 2π), n̄_range==(0, 0.5), θ_offset_range==(0, 0),
        point_dim==500, label_dim==70,
        "ci.jld2" in readdir(SqState.training_data_path())
    ])

    θs = LinRange(0, 2π, 10)
    xs = LinRange(-10, 10, 10)
    for i in 1:n
        r, θ, n̄, θ_offset = args[:, i]
        state = SqueezedThermalState(ξ(r, θ), n̄, dim=label_dim)

        ground_truth_pdf = q_pdf(state, θs, xs)
        sampled_pdf = pdf(kde((points[1, :, i], points[2, :, i])), θs, xs)
        @test sum(abs.(sampled_pdf .- ground_truth_pdf)) / n_points < 1e-3

        @test 𝛒s[i] == 𝛒(state)
    end
end

@testset "gen_non_gaussian_data" begin
    gen_non_gaussian_data()
end
