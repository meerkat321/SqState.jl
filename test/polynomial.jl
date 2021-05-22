function laguerre_horner(n::Integer, α::Integer, x::Real)
    # by Horner's method
    laguerre_l = 1
    bin = 1
    for i in n:-1:1
        bin *= (α + i) / (n + 1 - i)
        laguerre_l = bin - x * laguerre_l / i
    end

    return laguerre_l
end

laguerre_horner(n::Integer, α::Integer) = x->laguerre_horner(n, α, x)

@testset "laguerre" begin
    for n in 0:2:5, α in 0:2:5, x in -10:0.5:10
        # the precision of Horner's method is pretty bad when n is large
        thm_val = laguerre(n, α)(x)
        test_val = laguerre_horner(n, α)(x)
        @test abs(test_val - thm_val) < 1e-11
    end
end

@testset "hermite" begin
    for x in -10:0.5:10
        @test hermite(5)(x) ≈ 32x^5 - 160x^3 + 120x
    end
end
