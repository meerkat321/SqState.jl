using JLD2
using KernelDensity

export
    pdf,
    pdf!,
    gen_nongaussian_training_data,
    gen_gaussian_training_data,
    gen_gaussian_training_data!

#######
# pdf #
#######

real_tr_mul(𝐚, 𝐛) = sum(real(𝐚[i, :]' * 𝐛[:, i]) for i in 1:size(𝐚, 1))

function pdf(state::StateMatrix, θ::Real, x::Real; T=Float64)
    𝛑̂_res = Matrix{complex(T)}(undef, state.dim, state.dim)

    return pdf!(𝛑̂_res, state, θ, x)
end

function pdf!(𝛑̂_res::Matrix{Complex{T}}, state::StateMatrix, θ::Real, x::Real) where {T}
    if state.dim ≥ 455 && T != BigFloat
        @error "use `pdf(..., T=BigFloat)` if dimension of state is greater then 454"
        return 𝐩
    end

    return real_tr_mul(𝛑̂!(𝛑̂_res, T(θ), T(x), dim=state.dim), state.𝛒)
end

function pdf(state::StateMatrix, θs, xs; T=Float64)
    𝛑̂_res_vec = [Matrix{complex(T)}(undef, state.dim, state.dim) for _ in 1:Threads.nthreads()]
    𝐩 = Matrix{T}(undef, length(θs), length(xs))

    return pdf!(𝛑̂_res_vec, 𝐩, state, θs, xs)
end

function pdf!(𝛑̂_res_vec::Vector{Matrix{Complex{T}}}, 𝐩::Matrix{T}, state::StateMatrix, θs, xs) where {T}
    @sync for (j, x) in enumerate(xs)
        for (i, θ) in enumerate(θs)
            Threads.@spawn 𝐩[i, j] = pdf!(𝛑̂_res_vec[Threads.threadid()], state, θ, x)
        end
    end

    return 𝐩
end

##############################
# nongaussian data generator #
##############################
function ranged_rand(n, range::Tuple{T, T}) where {T <: Number}
    return range[1] .+ (range[2]-range[1]) * rand(T, n)
end

function ranged_rand(range::Tuple{T, T}) where {T <: Number}
    return range[1] + (range[2]-range[1]) * rand(T)
end

function gen_nongaussian_training_data(
    state::StateMatrix;
    n::Integer=4096, warm_up_n::Integer=128, batch_size=64,
    c=0.9, θ_range=(0., 2π), x_range=(-10., 10.),
    show_log=true
)
    sampled_points = Matrix{Float64}(undef, 2, n)
    𝛑̂_res_vec = [Matrix{complex(Float64)}(undef, state.dim, state.dim) for _ in 1:Threads.nthreads()]

    show_log && @info "Warm up"
    kde_result = kde((ranged_rand(n, θ_range), ranged_rand(n, x_range)))
    g = (θ, x) -> KernelDensity.pdf(kde_result, θ, x)
    Threads.@threads for i in 1:n # TODO: DEBUG
        sampled_points[:, i] .= [ranged_rand(θ_range), ranged_rand(x_range)]
        while SqState.pdf!(𝛑̂_res_vec[Threads.threadid()], state, sampled_points[:, i]...)/g(sampled_points[:, i]...)<c
            sampled_points[:, i] .= [ranged_rand(θ_range), ranged_rand(x_range)]
        end
    end

    # show_log && @info "Start to generate data"
    # batch = div(n-warm_up_n, batch_size)
    # for b in 1:batch
    #     ref_range = 1:(warm_up_n+(b-1)*batch_size)
    #     ref_points = view(sampled_points, :, ref_range)
    #     new_range = (warm_up_n+(b-1)*batch_size+1):(warm_up_n+b*batch_size)
    #     new_points = view(sampled_points, :, new_range)

    #     h = KernelDensity.default_bandwidth((ref_points[1, :], ref_points[2, :]))
    #     kde_result = kde((ref_points[1, :], ref_points[2, :]), bandwidth=h)
    #     g = (θ, x) -> KernelDensity.pdf(kde_result, θ, x)
    #     Threads.@threads for i in 1:batch_size
    #         new_points[:, i] .= ref_points[:, rand(ref_range)] + randn(2)./h
    #         while SqState.pdf!(𝛑̂_res_vec[Threads.threadid()], state, new_points[:, i]...)/g(new_points[:, i]...)<c || !(θ_range[1]≤new_points[1, i]≤θ_range[2])
    #             new_points[:, i] .= ref_points[:, rand(ref_range)] + randn(2)./h
    #         end
    #     end

    #     show_log && @info "progress: $b/$batch"
    # end

    return sampled_points[2, sortperm(sampled_points[1, :])]
end

###########################
# gaussian data generator #
###########################

function gen_gaussian_training_data(state::StateMatrix, n::Integer; bias_phase=0.)
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
    μ = π̂ₓ_μ(view(points, :), state)
    σ = real(sqrt.(π̂ₓ²_μ(view(points, :), state) - μ.^2))

    # xs
    view(points, :) .= real(μ) + σ .* randn(n)

    return points
end
