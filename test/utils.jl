@testset "factorial_ij" begin
    for i in 0:40, j in i:40
        @test SqState.factorial_ij(i, j) == factorial(big(i)) / factorial(big(j))
        @test SqState.factorial_ij(j, i) == factorial(big(i)) / factorial(big(j))
    end
end

@testset "z" begin
    x_range = -10:0.01:10
    p_range = -6:0.01:6

    for x in x_range, p in p_range
        @test SqState.z(x, p) == sqrt(2.)*(x + p*im)
    end

    xs = collect(x_range)
    ps = collect(p_range)
    @test SqState.z(xs, ps) == sqrt(2.).*(xs .+ ps' .* im)
end

@testset "gaussian_function" begin
    tol = 1e-14

    x_range = -10:0.01:10
    p_range = -6:0.01:6

    for x in x_range, p in p_range
        @test isapprox(SqState.gaussian_function(x, p), exp(-(x^2 + p^2)) / π, atol=tol)
    end

    xs = collect(x_range)
    ps = collect(p_range)
    @test isapprox(
        SqState.gaussian_function(xs, ps),
        exp.(-(xs.^2 .+ (ps').^2)) ./ π,
        atol=tol
    )
end

@testset "(-1)^i" begin
    for i in 1:50
        @test SqState.neg_one_to_power_of(i) == (-1)^i
    end
end

@testset "coefficient_of_wave_function" begin
    tol = 1e-14

    m_range = n_range = 1:70

    for m in m_range, n in n_range
        m1 = (n ≥ m) ? (-1)^(m-1) : (-1)^(n-1)
        @test isapprox(
            SqState.coefficient_of_wave_function(m, n),
            m1 * sqrt(factorial(big(min(m, n))) / factorial(big(max(m, n)))),
            atol=tol
        )
    end
end

@testset "z_to_power" begin
    tol = 1e-14

    m_range = n_range = 1:70
    x_range = -1:0.1:1
    p_range = -0.6:0.1:0.6

    for m in m_range, n in n_range, x in x_range, p in p_range
        z = (n >= m) ? sqrt(2.)*(x - p*im) : sqrt(2.)*(x + p*im)
        @test SqState.z_to_power(m, n, x, p) == z^(max(m, n) - min(m, n))
    end

    xs = collect(x_range)
    ps = collect(p_range)

    for m in m_range, n in n_range
        @test SqState.z_to_power(m, n, xs, ps) == SqState.z_to_power.(m, n, xs, ps')
    end
end

@testset "laguerre" begin
    m_range = n_range = 1:70
    x_range = -1:0.1:1
    p_range = -0.6:0.1:0.6

    # n >= m
    for m in m_range, n in m:n_range.stop, x in x_range, p in p_range
        @test abs(laguerre(m, n, x, p) - laguerre(m-1, n-m, abs2(sqrt(2.)*(x+p*im)))) < 1e-7
    end
    # n < m
    for n in n_range, m in n:m_range.stop, x in x_range, p in p_range
        @test abs(laguerre(m, n, x, p) - laguerre(n-1, m-n, abs2(sqrt(2.)*(x+p*im)))) < 1e-7
    end

    xs = collect(x_range)
    ps = collect(p_range)
    for m in m_range, n in n_range
        @test laguerre(m, n, xs, ps) == laguerre.(m, n, xs, ps')
    end
end
