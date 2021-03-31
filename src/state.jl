using LinearAlgebra

export
    AbstractState,
    purity,
    ρ,

    FockState,
    VacuumState,
    SinglePhotonState,
    NumberState,
    Arg,
    SuperpositionState

abstract type AbstractState end

purity(state::AbstractState) =  tr(ρ(state)^2)

struct FockState <: AbstractState
    n::Int64
end

VacuumState() = FockState(0)

SinglePhotonState() = FockState(1)

NumberState(n::Integer) = FockState(n)

function ρ(state::FockState; ρ_size=35)
    # rebase 0-based index system to 1-based
    n = state.n + 1

    ρ_fock = zeros(Complex, ρ_size, ρ_size)
    ρ_fock[n, n] = 1

    return ρ_fock
end

struct Arg
    r::Real
    θ::Real
end

z(arg::Arg) = arg.r * exp(im*arg.θ)

mutable struct SuperpositionState
    states::Vector{AbstractState}
    args::Vector{Arg}
end

SuperpositionState() = SuperpositionState(Vector{AbstractState}(), Vector{Arg}())

function Base.push!(
    superposition_state::SuperpositionState,
    state::AbstractState,
    arg::Arg
)
    push!(superposition_state.states, state)
    push!(superposition_state.args, arg)

    return superposition_state
end

function ρ(state::SuperpositionState; ρ_size=35)
    ρ_superposition = sum(z.(state.args) .* ρ.(state.states))
    ρ_superposition /= tr(ρ_superposition)

    return ρ_superposition
end
