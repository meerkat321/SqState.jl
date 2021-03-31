@testset "fock state" begin
    function test_fock_state(n)
        ρ = zeros(Complex, 35, 35)
        ρ[n+1, n+1] = 1
        @test fock_state(n) == ρ
    end

    for n in 0:34
        test_fock_state(n)
    end
end
