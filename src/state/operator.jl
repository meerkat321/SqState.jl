export
    Creation,
    create!,
    Annihilation,
    annihilate!,

    Arg,

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

Annihilation(; dim=DIM) = diagm(1 => sqrt.(1:dim-1))

function annihilate!(state::StateVector{<:Number})
    dim = state.dim
    state.v = Annihilation(dim=dim) * state.v

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
