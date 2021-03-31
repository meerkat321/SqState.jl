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
