@testset "fock state" begin
    function test_ρ_fock_state(n)
        state = FockState(n)

        ρ_fock = zeros(Complex, 35, 35)
        ρ_fock[n+1, n+1] = 1

        @test ρ(state) == ρ_fock
    end

    for n in 0:34
        test_ρ_fock_state(n)
    end
end
