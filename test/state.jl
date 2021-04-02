@testset "zero" begin
    state = Zero()
    @test vec(state) == 0
    @test ρ(state) == 0
    @test repr(state) == "0"

    @test annihilate(state) isa Zero
    @test create(state) isa Zero
    @test annihilateⁿ(state, 5) isa Zero
    @test createⁿ(state, 5) isa Zero
end

@testset "fock state" begin
    function test_ρ_fock_state(n)
        state = FockState(n)

        v_fock = zeros(Complex, 35)
        v_fock[n+1] = 1
        ρ_fock = zeros(Complex, 35, 35)
        ρ_fock[n+1, n+1] = 1

        @test vec(state) == v_fock
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
    fock_state = annihilate(fock_state)
    @test fock_state isa Zero

    fock_state = FockState(5)
    fock_state = annihilate(fock_state)
    @test fock_state.n == 4
    @test fock_state.w == sqrt(5)
    fock_state = annihilate(fock_state)
    @test fock_state.n == 3
    @test fock_state.w == sqrt(5) * sqrt(4)
    fock_state = annihilate(fock_state)
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

    @test createⁿ(SinglePhotonState(), 3).n == 4
    @test annihilateⁿ(FockState(5), 3).n == 2
    @test annihilateⁿ(FockState(5), 5).n == 0
    @test annihilateⁿ(FockState(5), 6) isa Zero
    @test annihilateⁿ(FockState(5), 7) isa Zero
    @test annihilateⁿ(FockState(5), 9) isa Zero
end

@testset "Arg" begin
    arg = Arg(2, π/4)
    @test SqState.z(arg) == arg.r * exp(-im*arg.θ)
    @test repr(arg) == "2.0 exp[-$(π/4)im]"
end

@testset "displacement" begin
    r = 2
    θ = π/4
    α = Arg(r, θ)

    α₀ = exp(-(abs(r)^2)/2)
    @test displacement(Arg(r, θ))(VacuumState()) == α₀ * sum([
        ComplexF64((α.r * exp(-im*α.θ))^n / factorial(big(n))) *
        vec(createⁿ(VacuumState(), n))
        for n in 0:35-1
    ])

end

@testset "Coherent State" begin
    r = 2
    θ = π/4
    coherent_state = CoherentState(Arg(r, θ))
    @test vec(coherent_state) == displacement(Arg(r, θ), dim=35)(VacuumState())
    @test ρ(coherent_state) == vec(coherent_state) * vec(coherent_state)'
    @test repr(coherent_state) == "D($(Arg(r, θ)))|0⟩"
end
