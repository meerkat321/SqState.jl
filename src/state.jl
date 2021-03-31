export
    AbstractState,
    FockState,
    ρ

abstract type AbstractState end

struct FockState <: AbstractState
    n::Int64
end

function ρ(state::FockState; ρ_size=35)
    # rebase 0-based index system to 1-based
    n = state.n + 1

    ρ_fock = zeros(Complex, ρ_size, ρ_size)
    ρ_fock[n, n] = 1

    return ρ_fock
end
