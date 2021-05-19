@testset "StateVector" begin
    dim = 35
    vacuum_state = FockState(ComplexF64, 0, dim=dim)

    state = zeros(ComplexF64, dim)
    state[0+1] = 1

    @test vacuum_state.v == state
    @test vacuum_state.dim == dim
end

@testset "FockState" begin
    vacuum_state = FockState(0)

    dim=35
    state = zeros(ComplexF64, dim)
    state[0+1] = 1

    @test vacuum_state.v == state
    @test vacuum_state.dim == dim

    vacuum_state = VacuumState()
    @test vacuum_state.v == state
    @test vacuum_state.dim == dim
end
