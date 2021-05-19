using LinearAlgebra

Base.:(==)(s1::StateVector, s2::StateVector) = (s1.v == s2.v) && (s1.dim == s2.dim)

@testset "StateVector" begin
    dim = 35
    vacuum_state = FockState(ComplexF64, 0, dim=dim)

    state = zeros(ComplexF64, dim)
    state[0+1] = 1

    @test vacuum_state.v == state
    @test vacuum_state.dim == dim

    @test repr(vacuum_state) == "StateVector{ComplexF64}( " *
        "\e[38;2;255;102;102m⬤" *
        "\e[38;2;178;178;178m⬤"^(dim-1) *
        "\e[0m )"
end

@testset "Alias" begin
    @test NumberState(0) == FockState(0)
    @test VacuumState() == FockState(0)
    @test SinglePhotonState() == FockState(1)
end

@testset "a and a†" begin
    dim = 35
    @test Creation() == diagm(-1 => sqrt.(1:dim-1))
    @test Annihilation() == diagm(1 => sqrt.(1:dim-1))

    @test create!(VacuumState()) == SinglePhotonState()
    @test annihilate!(SinglePhotonState()) == VacuumState()
end

@testset "Displacement" begin
    dim = 35

    @test repr(Arg(2., π/4)) == "Arg{Float64}(2.0exp($(π/4)im))"
    @test SqState.α(Arg(2., π/4)) == 2 * exp(im*π/4)

    @test Displacement(Arg(2., π/4)) == exp(
        2 * exp(im*π/4) * Creation(dim=dim) - 2 * exp(-im*π/4) * Annihilation(dim=dim)
    )
end

@testset "CoherentState" begin
    @test CoherentState(Arg(2., π/4)) == displace!(VacuumState(), Arg(2., π/4))
end

@testset "purity" begin
    @test purity(FockState(3)) == 1
    @test purity(VacuumState()) == 1
    @test purity(SinglePhotonState()) == 1
    @test purity(CoherentState(Arg(2., π/4))) == 1
end
