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
    createⁿ,
    displacement,
    CoherentState

abstract type AbstractState end

function purity(state::AbstractState)
    _ρ = ρ(state)
    _ρ /= tr(_ρ)

    return real(tr(_ρ^2))
end

struct Zero <: AbstractState end

Base.show(io::IO, ::Zero) = print(io, "0")

Base.vec(::Zero; dim=35) = 0

ρ(::Zero; dim=35) = 0

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

function Base.vec(state::FockState; dim=35)
    # rebase 0-based index system to 1-based
    n = state.n + 1

    v_fock = zeros(Complex, 35)
    v_fock[n] = state.w

    return v_fock
end

function ρ(state::FockState; dim=35)
    # rebase 0-based index system to 1-based
    n = state.n + 1

    ρ_fock = zeros(Complex, dim, dim)
    ρ_fock[n, n] = state.w

    return ρ_fock
end

struct Arg
    r::Float64
    θ::Float64
end

Base.show(io::IO, arg::Arg) = print(io, "$(arg.r) exp[-$(arg.θ)im]")

z(arg::Arg) = arg.r * exp(-im*arg.θ)

struct CoherentState <: AbstractState
    α::Arg
end

Base.show(io::IO, state::CoherentState) = print(io, "D($(state.α))|0⟩")

function createⁿ(state::FockState, n::Integer)
    for i in 1:n
        state = create(state)
    end

    return state
end

c(n::Integer, α::Arg) = ComplexF64(z(α)^n / factorial(big(n)))

function displacement(α::Arg; dim::Integer=35)
    α₀ = exp(-(abs(α.r)^2)/2)

    return (s::FockState) -> α₀ * sum([c(n, α) * vec(createⁿ(s, n)) for n in 0:dim-1])
end

Base.vec(state::CoherentState; dim=35) = displacement(state.α, dim=dim)(VacuumState())

function ρ(state::CoherentState; dim=35)
    coherent_state_vec = vec(state, dim=dim)
    return coherent_state_vec * coherent_state_vec'
end
