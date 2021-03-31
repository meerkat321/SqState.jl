using LinearAlgebra

export
    AbstractState,
    purity,
    ρ,

    Zero,

    FockState,
    VacuumState,
    SinglePhotonState,
    NumberState,
    destroy,
    create,

    Arg,
    SuperpositionState

abstract type AbstractState end

function purity(state::AbstractState)
    _ρ = ρ(state)
    _ρ /= tr(_ρ)

    return real(tr(_ρ^2))
end

struct Zero <: AbstractState end

Base.show(io::IO, zero::Zero) = print(io, "0")

ρ(zero::Zero; ρ_size=35) = 0

struct FockState <: AbstractState
    n::Int64
    w::ComplexF64
end

Base.show(io::IO, state::FockState) = print(io, "($(state.w))|$(state.n)⟩")

FockState(n::Integer) = FockState(n, 1)

VacuumState() = FockState(0)

SinglePhotonState() = FockState(1)

NumberState(n::Integer) = FockState(n)

function destroy(state::FockState)
    (state.n == 0) && (return Zero())
    return FockState(state.n-1, state.w*sqrt(state.n))
end

function create(state::FockState)
    return FockState(state.n+1, state.w*sqrt(state.n+1))
end

function ρ(state::FockState; ρ_size=35)
    # rebase 0-based index system to 1-based
    n = state.n + 1

    ρ_fock = zeros(Complex, ρ_size, ρ_size)
    ρ_fock[n, n] = state.w

    return ρ_fock
end

struct Arg
    r::Float64
    θ::Float64
end

Base.show(io::IO, arg::Arg) = print(io, "$(arg.r) exp[$(arg.θ)im]")

z(arg::Arg) = arg.r * exp(im*arg.θ)

mutable struct SuperpositionState <: AbstractState
    states::Vector{AbstractState}
    args::Vector{Arg}
end

function Base.show(io::IO, state::SuperpositionState)
    superposition_state_str = ""
    for (i, (w, s)) in enumerate(zip(state.args, state.states))
        (i != 1) && (superposition_state_str *= " + ")
        superposition_state_str *= "($w)$s"
    end

    print(io, superposition_state_str)
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

ρ(state::SuperpositionState; ρ_size=35) = sum(z.(state.args) .* ρ.(state.states))
