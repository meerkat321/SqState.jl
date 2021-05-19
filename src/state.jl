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

    StateMatrix,

    𝛒,
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
            round(Int, c.r * 255), round(Int, c.g * 255), round(Int, c.b * 255)
        )))\u2B24")
    end
    print(io, "$(Crayon(reset=true)) )")
end

Base.vec(state::StateVector{<:Number}) = state.v

𝛒(state::StateVector{<:Number}) = state.v * state.v'

function purity(state::StateVector{<:Number})
    𝛒 = state.v * state.v'
    𝛒 /= tr(𝛒)

    return real(tr(𝛒^2))
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

const ξ = α

function Squeezing(arg::Arg{<:Real}; dim=DIM)
    return exp(0.5 * ξ(arg)' * Annihilation(dim=dim)^2 - 0.5 * ξ(arg) * Creation(dim=dim)^2)
end

function squeeze!(state::StateVector{<:Number}, arg::Arg{<:Real})
    dim = state.dim
    state.v = Squeezing(arg, dim=dim) * state.v

    return state
end

function SqueezedState(arg::Arg{<:Real}; dim=DIM)
    return squeeze!(VacuumState(dim=dim), arg)
end

mutable struct StateMatrix{T <: Number} <: AbstractState
    𝛒::Matrix{T}
    dim::Int64
end

function Base.show(io::IO, state::StateMatrix{T}) where {T}
    function show_𝛒(𝛒::Matrix{<:Real})
        for (i, p) in enumerate(𝛒)
            c = (p>0) ? convert(RGB, HSL(0, p, 0.7)) : convert(RGB, HSL(240, abs(p), 0.7))
            print(io, "    $(Crayon(foreground=(
                round(Int, c.r * 255), round(Int, c.g * 255), round(Int, c.b * 255)
            )))\u2B24")
            (i%state.dim == 0) && println(io)
        end
    end

    println(io, "StateMatrix{$T}(")
    𝛒_r = real(state.𝛒)
    𝛒_r /= maximum(abs.(𝛒_r))
    show_𝛒(𝛒_r)
    print(io, "$(Crayon(reset=true)))")
end

function StateMatrix(state::StateVector{T}) where {T <: Number}
    𝛒 = state.v * state.v'
    return StateMatrix{T}(𝛒, state.dim)
end

𝛒(state::StateMatrix{<:Number}) = state.𝛒

function purity(state::StateMatrix{<:Number})
    𝛒 = state.𝛒
    𝛒 /= tr(𝛒)

    return real(tr(𝛒^2))
end
