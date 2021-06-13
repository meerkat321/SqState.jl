using TransformVariables
using LogDensityProblems
using DynamicHMC
using Parameters
using Statistics
using Random
using ForwardDiff

export
    pdf_θ_x,
    gen_data

function pdf_θ_x(state::StateMatrix, θ::Real, x::Real)
    return real(tr(𝛑_θ_x(θ, x, dim=state.dim) * state.𝛒))
end

struct QuantumStateProblem
    state::StateMatrix
end

function (problem::QuantumStateProblem)(𝐱)
    @unpack θ, x = 𝐱
    @unpack state = problem
    p = pdf_θ_x(state, θ, x)
    p = (p <= 0) ? floatmin() : p

    return log(p)
end

function gen_data(state::StateMatrix; n::Integer=40960, θ_range::Tuple=(0., 2π), x_range=(-20., 20.))
    second = arr -> arr[2]
    t = as((θ=as(Real,θ_range[1], θ_range[2]), x=as(Real, x_range[1], x_range[2])))

    problem = QuantumStateProblem(state)

    log_likelyhood = TransformedLogDensity(t, problem)
    ∇log_likelyhood = ADgradient(:ForwardDiff, log_likelyhood)

    results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇log_likelyhood, n)
    sampled_data = transform.(t, results.chain)

    return hcat(first.(sampled_data), second.(sampled_data))
end
