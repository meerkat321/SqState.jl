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
function ranged_rand(n, range::Tuple{T, T}) where {T <: Number}
    return range[1] .+ (range[2]-range[1]) * rand(T, n)
end

ranged_rand(range) = ranged_rand(1, range)[1]

is_rejected(point, p, g, c) = p(point...) / g(point...) < c

function gen_warm_up_point(p, g, c, θ_range, x_range)
    new_point = Vector{Float64}(undef, 2)

    return gen_warm_up_point!(new_point, p, g, c, θ_range, x_range)
end

function gen_warm_up_point!(new_point, p, g, c, θ_range, x_range)
    new_point .= [ranged_rand(θ_range), ranged_rand(x_range)]
    while is_rejected(new_point, p, g, c)
        new_point .= [ranged_rand(θ_range), ranged_rand(x_range)]
    end

    return new_point
end

function warm_up(n, p, g, c, θ_range, x_range)
    points = Matrix{Float64}(undef, 2, n)

    return warm_up!(points, n, p, g, c, θ_range, x_range)
end

function warm_up!(points, n, p, g, c, θ_range, x_range)
    sp_lock = Threads.SpinLock()
    Threads.@threads for i in 1:n
        new_point = Vector{Float64}(undef, 2)
        gen_warm_up_point!(new_point, p, g, c, θ_range, x_range)

        lock(sp_lock) do
            view(points, :, i) .= new_point
        end
    end

    return points
end

function gen_point(sampled_points, p, g, c, h, θ_range, x_range)
    new_point = Vector{Float64}(undef, 2)

    return gen_point!(new_point, sampled_points, p, g, c, h, θ_range, x_range)
end

function gen_point!(new_point, sampled_points, p, g, c, h, θ_range, x_range)
    ref_range = 1:size(sampled_points, 2)

    new_point .= view(sampled_points, :, rand(ref_range)) + randn(2)./h
	while is_rejected(new_point, p, g, c) || !(θ_range[1]≤new_point[1]≤θ_range[2])
        new_point .= view(sampled_points, :, rand(ref_range)) + randn(2)./h
	end

    return new_point
end

function gen_fragment_nongaussian_data(sampled_points, n, p, g, c, h, θ_range, x_range)
    points = Matrix{Float64}(undef, 2, n)

    return gen_fragment_nongaussian_data!(points, sampled_points, n, p, g, c, h, θ_range, x_range)
end

function gen_fragment_nongaussian_data!(points, sampled_points, n, p, g, c, h, θ_range, x_range)
    sp_lock = Threads.SpinLock()
    Threads.@threads for i in 1:n
        new_point = Vector{Float64}(undef, 2)
        gen_point!(new_point, sampled_points, p, g, c, h, θ_range, x_range)

        lock(sp_lock) do
            view(points, :, i) .= new_point
        end
    end

    return points
end

function gen_nongaussian_training_data(
    state::StateMatrix;
    n::Integer=4096, warm_up_n::Integer=128, batch_size=64,
    c=0.9, θ_range=(0., 2π), x_range=(-10., 10.),
    show_log=true
)
    sampled_points = Matrix{Float64}(undef, 2, n)

    p = (θ, x) -> SqState.pdf(state, θ, x)

    show_log && @info "Warm up"
    kde_result = kde((ranged_rand(n, θ_range), ranged_rand(n, x_range)))
    g = (θ, x) -> KernelDensity.pdf(kde_result, θ, x)
    warm_up!(view(sampled_points, :, 1:warm_up_n), warm_up_n, p, g, c, θ_range, x_range)

    show_log && @info "Start to generate data"
    batch = div(n-warm_up_n, batch_size)
    for i in 1:batch
        h = KernelDensity.default_bandwidth((
            view(sampled_points, 1, 1:(warm_up_n+(i-1)*batch_size)),
            view(sampled_points, 2, 1:(warm_up_n+(i-1)*batch_size))
        ))
        kde_result = kde(
            (
                view(sampled_points, 1, 1:(warm_up_n+(i-1)*batch_size)),
                view(sampled_points, 2, 1:(warm_up_n+(i-1)*batch_size))
            ),
            bandwidth=h
        )
        g = (θ, x) -> KernelDensity.pdf(kde_result, θ, x)

        gen_fragment_nongaussian_data!(
            view(sampled_points, :, (warm_up_n+(i-1)*batch_size+1):(warm_up_n+(i)*batch_size)),
            view(sampled_points, :, 1:(warm_up_n+(i-1)*batch_size)),
            batch_size, p, g, c, h, θ_range, x_range
        )

        show_log && @info "progress: $i/$batch"
    end

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
