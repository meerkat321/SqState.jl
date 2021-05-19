using LinearAlgebra

export
    FockState,
    NumberState,
    VacuumState,
    SinglePhotonState,

    Creation,
    Annihilation,

    Arg,
    Displacement,
    displace!,
    CoherentState,
    purity

abstract type AbstractState end

mutable struct StateVector{T <: Number} <: AbstractState
    v::Vector{T}
    dim::Int64
end

function FockState(T::Type{<:Number}, n::Integer; dim::Integer)
    v = zeros(T, dim)
    v[n+1] = 1

    return StateVector{T}(v, dim)
end

FockState(n; dim=DIM) = FockState(ComplexF64, n, dim=dim)

NumberState(n; dim=DIM) = FockState(ComplexF64, n, dim=dim)

VacuumState(; dim=DIM) = FockState(ComplexF64, 0, dim=dim)

SinglePhotonState(; dim=DIM) = FockState(ComplexF64, 1, dim=dim)

Creation(; dim=DIM) = diagm(-1 => sqrt.(1:dim-1))

Annihilation(; dim=DIM) = diagm(1 => sqrt.(1:dim-1))

struct Arg{T <: Real}
    r::T
    θ::T
end

α(arg::Arg{<:Real}) = arg.r * exp(im * arg.θ)

function Displacement(arg::Arg{<:Real}; dim=DIM)
    return exp(α(arg) * Creation(dim=dim) - α(arg)' * Annihilation(dim=dim))
end

function displace!(state::StateVector, arg::Arg{<:Real})
    dim = state.dim
    state.v = Displacement(arg, dim=dim) * state.v

    return state
end

function CoherentState(arg::Arg{<:Real}; dim=DIM)
    return displace!(VacuumState(dim=dim), arg)
end

function putity(state::StateVector)
    ρ = state.v * state.v'
    ρ /= (ρ)

    return real(tr(ρ))
end
