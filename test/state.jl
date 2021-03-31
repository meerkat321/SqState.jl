@testset "zero" begin
    state = Zero()
    @test ρ(state) == 0
    @test repr(state) == "0"
end

@testset "fock state" begin
    function test_ρ_fock_state(n)
        state = FockState(n)

        ρ_fock = zeros(Complex, 35, 35)
        ρ_fock[n+1, n+1] = 1

        @test ρ(state) == ρ_fock
        @test purity(state) == 1
        @test repr(state) == "(1.0 + 0.0im)|$n⟩"
    end

    for n in 0:34
        test_ρ_fock_state(n)
    end

    @test VacuumState() == FockState(0)
    @test SinglePhotonState() == FockState(1)
    @test NumberState(5) == FockState(5)
end

@testset "a and a†" begin
    fock_state = VacuumState()
    fock_state = destroy(fock_state)
    @test fock_state isa Zero

    fock_state = FockState(5)
    fock_state = destroy(fock_state)
    @test fock_state.n == 4
    @test fock_state.w == sqrt(5)
    fock_state = destroy(fock_state)
    @test fock_state.n == 3
    @test fock_state.w == sqrt(5) * sqrt(4)
    fock_state = destroy(fock_state)
    @test fock_state.n == 2
    @test fock_state.w == sqrt(5) * sqrt(4) * sqrt(3)


    fock_state = VacuumState()
    fock_state = create(fock_state)
    @test fock_state.n == 1
    @test fock_state.w == sqrt(1)

    fock_state = create(fock_state)
    @test fock_state.n == 2
    @test fock_state.w == sqrt(1) * sqrt(2)

    fock_state = create(fock_state)
    @test fock_state.n == 3
    @test fock_state.w == sqrt(1) * sqrt(2) * sqrt(3)
end

@testset "Arg" begin
    arg = Arg(2, π/4)
    @test SqState.z(arg) == arg.r * exp(im*arg.θ)
    @test repr(arg) == "2.0 exp[$(π/4)im]"
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
                    (arg0.r * exp(im*arg0.θ)) * 1
            elseif (row == 2+1 && col == 2+1)
                @test ρ_superposition[row, col] ==
                    (arg2.r * exp(im*arg2.θ)) * 1
            else
                @test ρ_superposition[row, col] == 0
            end
        end
    end

    @test purity(superposition_state) == real(
        ((arg0.r * exp(im*arg0.θ)) * 1 / normalize_c)^2 +
        ((arg2.r * exp(im*arg2.θ)) * 1 / normalize_c)^2
    )
    @test repr(superposition_state) == "($arg0)(1.0 + 0.0im)|0⟩ + ($arg2)(1.0 + 0.0im)|2⟩"
end
