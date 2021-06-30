using JLD2
using TransformVariables
using LogDensityProblems
using DynamicHMC
using Parameters
using Statistics
using Random
using ForwardDiff
using KernelDensity

export
    pdf,
    pdf!,
    DHMC,
    Rejection,
    gen_nongaussian_training_data,
    gen_gaussian_training_data,
    gen_gaussian_training_data!

#######
# pdf #
#######

real_tr_mul(𝐚, 𝐛) = sum(real(𝐚[i, :]' * 𝐛[:, i]) for i in 1:size(𝐚, 1))

function pdf(state::StateMatrix, θ::Real, x::Real)
    return real_tr_mul(𝛑̂(θ, x, dim=state.dim), state.𝛒)
end

function pdf(state::StateMatrix, θs, xs; T=Float64)
    𝐩 = Matrix{T}(undef, length(θs), length(xs))

    return pdf!(𝐩, state, θs, xs)
end

function pdf!(𝐩::Matrix{T}, state::StateMatrix, θs, xs) where {T}
    𝛑̂_res = Matrix{complex(T)}(undef, state.dim, state.dim)

    for (j, x) in enumerate(xs)
        for (i, θ) in enumerate(θs)
            𝐩[i, j] = real_tr_mul(𝛑̂!(𝛑̂_res, θ, x; dim=state.dim), state.𝛒)
        end
    end

    return 𝐩
end

##############################
# nongaussian data generator #
##############################

abstract type AbstractSamplingMethod end

struct DHMC <: AbstractSamplingMethod end

struct Rejection <: AbstractSamplingMethod end

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

function gen_nongaussian_training_data(
    state::StateMatrix, ::Type{DHMC};
    n::Integer=40960, θ_range::Tuple=(0., 2π), x_range=(-10., 10.)
)
    second = arr -> arr[2]
    t = as((θ=as(Real, θ_range...), x=as(Real, x_range...)))

    problem = QuantumStateProblem(state)

    log_likelyhood = TransformedLogDensity(t, problem)
    ∇log_likelyhood = ADgradient(:ForwardDiff, log_likelyhood)

    results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇log_likelyhood, n)
    sampled_data = transform.(t, results.chain)

    return hcat(first.(sampled_data), second.(sampled_data)), results
end

function rand2range(rand::T, range::Tuple{T, T}) where {T <: Number}
    return range[1] + (range[2]-range[1]) * rand
end

function rand2range(rand::Vector{T}, range::Tuple{T, T}) where {T <: Number}
    return range[1] .+ (range[2]-range[1]) * rand
end

function accept_reject(p, g, c, θ_range, x_range)
    new_data = Vector{Float64}(undef, 2)

    return accept_reject!(new_data, p, g, c, θ_range, x_range)
end

function accept_reject!(new_data::Vector, p, g, c, θ_range, x_range)
    view(new_data, :) .= [
        rand2range(rand(),θ_range),
        rand2range(rand(), x_range)
    ]
    while p(new_data...) / g(new_data...) < c
        view(new_data, :) .= [
            rand2range(rand(),θ_range),
            rand2range(rand(), x_range)
        ]
    end

    return new_data
end

function gen_batch_nongaussian_training_data!(data::SubArray, p, g, c, θ_range, x_range)
    n = size(data, 1)

    sp_lock = Threads.SpinLock()
    Threads.@threads for i in 1:n
        new_data = Vector{Float64}(undef, 2)
        accept_reject!(new_data, p, g, c, θ_range, x_range)

        lock(sp_lock) do
            data[i, :] .= new_data
        end
    end

    return data
end

function gen_nongaussian_training_data(
    state::StateMatrix, ::Type{Rejection};
    n::Integer=4096, batch_size=32, c=0.9, θ_range=(0., 2π), x_range=(-10., 10.)
)
    data = Matrix{Float64}(undef, n, 2)
    p = (θ, x) -> SqState.pdf(state, θ, x)

    @info "Initial g"
    kde_result = kde((rand2range(rand(n),θ_range), rand2range(rand(n), x_range)))
    g = (θ, x) -> KernelDensity.pdf(kde_result, θ, x)
    gen_batch_nongaussian_training_data!(view(data, 1:batch_size, :), p, g, c, θ_range, x_range)
    kde_result = kde((data[1:batch_size, 1], data[1:batch_size, 2]))
    g = (θ, x) -> KernelDensity.pdf(kde_result, θ, x)

    @info "Start to generate data"
    batch = div(n, batch_size)
    for i in 1:batch
        gen_batch_nongaussian_training_data!(view(data, (i-1)*batch_size+1:i*batch_size, :), p, g, c, θ_range, x_range)
        kde_result = kde((data[1:i*batch_size, 1], data[1:i*batch_size, 2]))
        g = (θ, x) -> KernelDensity.pdf(kde_result, θ, x)
        @info "progress: $i/$batch"
    end

    return data, kde_result
end

###########################
# gaussian data generator #
###########################

function gen_gaussian_training_data(state::StateMatrix, n::Integer; bias_phase=0)
    points = Vector{Float64}(undef, n)

    return gen_gaussian_training_data!(points, state, bias_phase)
end

function gen_gaussian_training_data!(
    points::AbstractVector{Float64},
    state::StateMatrix, bias_phase::Float64
)
    n = length(points)

    # θs
    view(points, :) .= sort!(2π*rand(n) .+ bias_phase)

    # μ and σ given θ
    μ = Δπ̂ₓ(view(points, :), state)
    σ = real(sqrt.(Δπ̂ₓ²(view(points, :), state) - μ.^2))

    # xs
    view(points, :) .= real(μ) + σ .* randn(n)

    return points
end
