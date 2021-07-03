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
function rand2range(rand::T, range::Tuple{T, T}) where {T <: Number}
    return range[1] + (range[2]-range[1]) * rand
end

function rand2range(rand::Vector{T}, range::Tuple{T, T}) where {T <: Number}
    return range[1] .+ (range[2]-range[1]) * rand
end

is_rejected(point, p, g, c) = p(point...) / g(point...) < c

function gen_warm_up_point(p, g, c, θ_range, x_range)
    new_data = Vector{Float64}(undef, 2)

    return gen_warm_up_point!(new_data, p, g, c, θ_range, x_range)
end

function gen_warm_up_point!(new_data::Vector, p, g, c, θ_range, x_range)
    view(new_data, :) .= [
        rand2range(rand(),θ_range),
        rand2range(rand(), x_range)
    ]
    while is_rejected(new_data, p, g, c)
        view(new_data, :) .= [
            rand2range(rand(),θ_range),
            rand2range(rand(), x_range)
        ]
    end

    return new_data
end

function warm_up!(data, n, p, g, c, θ_range, x_range)
	sp_lock = Threads.SpinLock()
    Threads.@threads for i in 1:n
        new_data = Vector{Float64}(undef, 2)
        gen_warm_up_point!(new_data, p, g, c, θ_range, x_range)

        lock(sp_lock) do
            view(data, i, :) .= new_data
        end
    end
end

function gen_point!(new_data::Vector, current_points, p, g, c, h, θ_range, x_range)
    i = rand(1:size(current_points, 1))

	view(new_data, :) .= current_points[i, :] + 2rand(2).-1
	while !(θ_range[1]<new_data[1]<θ_range[2]) || is_rejected(new_data, p, g, c)
	    view(new_data, :) .= current_points[i, :] + (1 ./ h) .* randn(2)
	end

    return new_data
end

function gen_batch_nongaussian_training_data!(
    data, ref_range, fill_range,
    p, g, c, h, θ_range, x_range
)
    sp_lock = Threads.SpinLock()
    Threads.@threads for i in fill_range
        new_data = Vector{Float64}(undef, 2)
        gen_point!(new_data, view(data, ref_range, :), p, g, c, h, θ_range, x_range)

        lock(sp_lock) do
            view(data, i, :) .= new_data
        end
    end

    return data
end

function gen_nongaussian_training_data(
    state::StateMatrix;
    n::Integer=4096, batch_size=64, c=0.9, θ_range=(0., 2π), x_range=(-10., 10.),
    show_log=true
)
    data = Matrix{Float64}(undef, n, 2)
    p = (θ, x) -> SqState.pdf(state, θ, x)

    show_log && @info "Initial g"
    kde_result = kde((rand2range(rand(n),θ_range), rand2range(rand(n), x_range)))
    g = (θ, x) -> KernelDensity.pdf(kde_result, θ, x)
    warm_up!(data, batch_size, p, g, c, θ_range, x_range)

    show_log && @info "Start to generate data"
    batch = div(n, batch_size)
    for i in 2:batch
        h = KernelDensity.default_bandwidth((data[1:(i-1)*batch_size, 1], data[1:(i-1)*batch_size, 2]))
        kde_result = kde((data[1:(i-1)*batch_size, 1], data[1:(i-1)*batch_size, 2]), bandwidth=h)
        g = (θ, x) -> KernelDensity.pdf(kde_result, θ, x)
        gen_batch_nongaussian_training_data!(
			data, 1:(i-1)*batch_size, (i-1)*batch_size.+(1:batch_size),
			p, g, c, h, θ_range, x_range
		)
        show_log && @info "progress: $i/$batch"
    end

    data .= data[sortperm(data[:, 1]), :]

    return data
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
