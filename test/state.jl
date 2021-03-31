@testset "fock state" begin
    function test_ρ_fock_state(n)
        state = FockState(n)

        ρ_fock = zeros(Complex, 35, 35)
        ρ_fock[n+1, n+1] = 1

        @test ρ(state) == ρ_fock
        @test purity(state) == 1
    end

    for n in 0:34
        test_ρ_fock_state(n)
    end

    @test VacuumState() == FockState(0)
end

@testset "superposition state" begin
    s0 = FockState(0)
    arg0 = Arg(2, π/4)
    s2 = FockState(2)
    arg2 = Arg(5, π/4)

    superposition_state = SuperpositionState()
    push!(superposition_state, s0, arg0)
    push!(superposition_state, s2, arg2)

    ρ_superposition = ρ(superposition_state)

    normalize_c = (arg0.r * exp(im*arg0.θ))*1 + (arg2.r * exp(im*arg2.θ))*1

    for row in 1:35
        for col in 1:35
            if (row == 0+1 && col == 0+1)
                @test ρ_superposition[row, col] ==
                    (arg0.r * exp(im*arg0.θ)) * 1 / normalize_c
            elseif (row == 2+1 && col == 2+1)
                @test ρ_superposition[row, col] ==
                    (arg2.r * exp(im*arg2.θ)) * 1 / normalize_c
            else
                @test ρ_superposition[row, col] == 0
            end
        end
    end
end
