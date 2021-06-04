export
    Creation,
    create!,
    Annihilation,
    annihilate!,

    Arg,
    α,
    ξ,

    Displacement,
    displace!,

    Squeezing,
    squeeze!

############
# a† and a #
############

Creation(; dim=DIM) = diagm(-1 => sqrt.(1:dim-1))

function create!(state::StateVector{<:Number})
    dim = state.dim
    state.v = Creation(dim=dim) * state.v

    return state
end

function create!(state::StateMatrix{<:Number})
    dim = state.dim
    𝐜 = Creation(dim=dim)
    state.𝛒 = 𝐜 * state.𝛒 * 𝐜'

    return state
end

Annihilation(; dim=DIM) = diagm(1 => sqrt.(1:dim-1))

function annihilate!(state::StateVector{<:Number})
    dim = state.dim
    state.v = Annihilation(dim=dim) * state.v

    return state
end

function annihilate!(state::StateMatrix{<:Number})
    dim = state.dim
    𝐚 = Annihilation(dim=dim)
    state.𝛒 = 𝐚 * state.𝛒 * 𝐚'

    return state
end

###########
# α and ξ #
###########

struct Arg{T <: Real}
    r::T
    θ::T
end

Base.show(io::IO, arg::Arg{T}) where {T} = print(io, "Arg{$T}($(arg.r)exp($(arg.θ)im))")

z(arg::Arg{<:Real}) = arg.r * exp(im * arg.θ)

α(r::T, θ::T) where {T} = Arg{T}(r, θ)
const ξ = α

################
# displacement #
################

function Displacement(α::Arg{<:Real}; dim=DIM)
    return exp(z(α) * Creation(dim=dim) - z(α)' * Annihilation(dim=dim))
end

function displace!(state::StateVector{<:Number}, α::Arg{<:Real})
    dim = state.dim
    state.v = Displacement(α, dim=dim) * state.v

    return state
end

function displace!(state::StateMatrix{<:Number}, α::Arg{<:Real})
    dim = state.dim
    𝐝 = Displacement(α, dim=dim)
    state.𝛒 = 𝐝 * state.𝛒 * 𝐝'

    return state
end

#############
# squeezing #
#############

function Squeezing(ξ::Arg{<:Real}; dim=DIM)
    return exp(0.5 * z(ξ)' * Annihilation(dim=dim)^2 - 0.5 * z(ξ) * Creation(dim=dim)^2)
end

function squeeze!(state::StateVector{<:Number}, ξ::Arg{<:Real})
    dim = state.dim
    state.v = Squeezing(ξ, dim=dim) * state.v

    return state
end

function squeeze!(state::StateMatrix{<:Number}, ξ::Arg{<:Real})
    dim = state.dim
    𝐬 = Squeezing(ξ, dim=dim)
    state.𝛒 = 𝐬 * state.𝛒 * 𝐬'

    return state
end

###############
# measurement #
###############

# |θ, x⟩ = ∑ₙ |n⟩ ⟨n|θ, x⟩ = ∑ₙ ψₙ(θ, x) |n⟩
# ⟨n|θ, x⟩ = ψₙ(θ, x) = exp(im n θ) (2/π)^(1/4) exp(-x^2) Hₙ(√2 x)/√(2^n n!)
function ψₙ_θ_x(n::Integer, θ::Real, x::Real)
    return exp(im * n * θ) *
        (2/π) ^ (1/4) *
        exp(-x^2) *
        hermite(n)(sqrt(2)x) / sqrt(2^n * factorial(n))
end

function 𝛑_θ_x(; dim=big(DIM))
    return (θ, x) -> ψₙ_θ_x.(0:dim-1, θ, x) * ψₙ_θ_x.(0:dim-1, θ, x)'
end

function prob_θ_x(state::StateMatrix)
    dim = (state.dim>20) ? big(state.dim) : state.dim

    return (θ, x) -> real(tr(𝛑_θ_x(dim=dim)(θ, x) * state.𝛒))
end

###########

function ψₙ_θ(n::Integer, θ::Real)
    return exp(im * n * θ)
end

function ψₙ_x(n::Integer, x::Real)
    return (2/π) ^ (1/4) * exp(-x^2) * hermite(n)(sqrt(2)x) / sqrt(2^n * factorial(n))
end

function 𝛑_θ(; dim=big(DIM))
    return θ -> ψₙ_θ.(0:dim-1, θ) * ψₙ_θ.(0:dim-1, θ)'
end

function 𝛑_x(; dim=big(DIM))
    return x -> ψₙ_x.(0:dim-1, x) * ψₙ_x.(0:dim-1, x)'
end

function prob_θ(state::StateMatrix)
    dim = (state.dim>20) ? big(state.dim) : state.dim

    return θ -> real(tr(𝛑_θ(dim=dim)(θ) * state.𝛒))
end

function prob_x(state::StateMatrix)
    dim = (state.dim>20) ? big(state.dim) : state.dim

    return x -> real(tr(𝛑_x(dim=dim)(x) * state.𝛒))
end
