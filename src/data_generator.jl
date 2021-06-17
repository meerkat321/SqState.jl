using JLD2
using TransformVariables
using LogDensityProblems
using DynamicHMC
using Parameters
using Statistics
using Random
using ForwardDiff
using Distributions

export
    pdf,
    pdf!,
    gen_nongaussian_training_data,
    gen_gaussian_training_data

real_tr_mul(𝐚, 𝐛) = sum(real(𝐚[i, :]' * 𝐛[:, i]) for i in 1:size(𝐚, 1))

function pdf(state::StateMatrix, θ::Real, x::Real)
    return real_tr_mul(𝛑(θ, x, dim=state.dim), state.𝛒)
end

function pdf(state::StateMatrix, θs, xs; T=Float64)
    𝐩 = Matrix{T}(undef, length(θs), length(xs))

    return pdf!(𝐩, state, θs, xs)
end

function pdf!(𝐩::Matrix{T}, state::StateMatrix, θs, xs) where {T}
    𝛑_res = Matrix{complex(T)}(undef, state.dim, state.dim)

    for (j, x) in enumerate(xs)
        for (i, θ) in enumerate(θs)
            𝐩[i, j] = real_tr_mul(𝛑!(𝛑_res, θ, x; dim=state.dim), state.𝛒)
        end
    end

    return 𝐩
end

struct QuantumStateProblem
    state::StateMatrix
end

function (problem::QuantumStateProblem)(𝐱)
    @unpack θ, x = 𝐱
    @unpack state = problem

    ψₙs = ψₙ.(0:state.dim-1, θ, x)
    p = real_tr_mul(ψₙs*ψₙs', state.𝛒)
    p = (p <= 0) ? floatmin() : p

    return log(p)
end

function gen_nongaussian_training_data(state::StateMatrix; n::Integer=40960, θ_range::Tuple=(0., 2π), x_range=(-20., 20.))
    second = arr -> arr[2]
    t = as((θ=as(Real,θ_range[1], θ_range[2]), x=as(Real, x_range[1], x_range[2])))

    problem = QuantumStateProblem(state)

    log_likelyhood = TransformedLogDensity(t, problem)
    ∇log_likelyhood = ADgradient(:ForwardDiff, log_likelyhood)

    results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇log_likelyhood, n)
    sampled_data = transform.(t, results.chain)

    return hcat(first.(sampled_data), second.(sampled_data)), results
end

function gen_gaussian_training_data(state::StateMatrix, n::Integer)
    a = tr(annihilate(state).𝛒)
    a² = tr(annihilate!(annihilate(state)).𝛒)
    ad = tr(create(state).𝛒)
    ad² = tr(create!(create(state)).𝛒)
    ada = tr(create!(annihilate(state)).𝛒)

    θs = 2π * rand(n)

    q² = @. 0.5(a²*exp(-2im*θs) + ad²*exp(2im*θs) + 1 + 2ada)
    μ = 1/sqrt(2) * (a*exp.(-im*θs) + ad*exp.(im*θs))
    σ² = real((q² - μ.^2) / 2)
    σ² = map(x->(x≤0 ? floatmin() : x), σ²)

    xs = μ + sqrt.(σ²) .* randn(n)

    return hcat(θs, xs)
end
