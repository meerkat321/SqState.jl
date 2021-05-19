using LinearAlgebra
using Crayons

export
    StateVector,
    FockState,
    NumberState,
    VacuumState,
    SinglePhotonState,

    Creation,
    create!,
    Annihilation,
    annihilate!,

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

function Base.show(io::IO, state::StateVector{T}) where {T}
    print(io, "StateVector{$T}( ")
    v = abs2.(state.v)
    v /= maximum(v)
    for p in v
        c = convert(RGB, HSL(0, p, 0.7))
        print(io, "$(Crayon(foreground=(
                round(Int, c.r * 255),
                round(Int, c.g * 255),
                round(Int, c.b * 255)
            )))\u2B24"
        )
    end
    print(io, "$(Crayon(reset=true)) )")
end

function purity(state::StateVector{<:Number})
    ρ = state.v * state.v'
    ρ /= tr(ρ)

    return real(tr(ρ^2))
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

function create!(state::StateVector{<:Number})
    dim = state.dim
    state.v = Creation(dim=dim) * state.v

    return state
end

Annihilation(; dim=DIM) = diagm(1 => sqrt.(1:dim-1))

function annihilate!(state::StateVector{<:Number})
    dim = state.dim
    state.v = Annihilation(dim=dim) * state.v

    return state
end

struct Arg{T <: Real}
    r::T
    θ::T
end

Base.show(io::IO, arg::Arg{T}) where {T} = print(io, "Arg{$T}($(arg.r)exp($(arg.θ)im))")

α(arg::Arg{<:Real}) = arg.r * exp(im * arg.θ)

function Displacement(arg::Arg{<:Real}; dim=DIM)
    return exp(α(arg) * Creation(dim=dim) - α(arg)' * Annihilation(dim=dim))
end

function displace!(state::StateVector{<:Number}, arg::Arg{<:Real})
    dim = state.dim
    state.v = Displacement(arg, dim=dim) * state.v

    return state
end

function CoherentState(arg::Arg{<:Real}; dim=DIM)
    return displace!(VacuumState(dim=dim), arg)
end
