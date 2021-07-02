### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ bb7cc2c4-db59-11eb-09a5-fbde282f71e6
begin
	using BenchmarkTools
	using SqState
	using KernelDensity
	using Plots
end

# ╔═╡ b01fbeb1-ad18-4788-becb-c97c6421038f
begin
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
end

# ╔═╡ 0c700e42-99d5-4308-b63a-966c332fca6a
function gen_point(current_points, θ_range, x_range)
	h = KernelDensity.default_bandwidth((current_points[:, 1], current_points[:, 2]))
	i = rand(1:size(current_points, 1))

	new_data = current_points[i, :] + 2rand(2).-1
	while !(θ_range[1]<new_data[1]<θ_range[2])
		new_data .= current_points[i, :] + (1 ./ h) .* randn(2)
	end

	return new_data
end

# ╔═╡ 22b8d28a-7beb-4af3-83a8-482bed72bead
function gen_batch_nongaussian_training_data!(
		data, ref_range, fill_range,
		p, g, c, θ_range, x_range
)
    sp_lock = Threads.SpinLock()
    Threads.@threads for i in fill_range
		new_data = gen_point(view(data, ref_range, :), θ_range, x_range)
		while p(new_data...) / g(new_data...) < c
			new_data .= gen_point(view(data, ref_range, :), θ_range, x_range)
		end

        lock(sp_lock) do
            view(data, i, :) .= new_data
        end
    end

    return data
end

# ╔═╡ 54d609b8-58f0-4f1e-aa1b-1ac6319184b9
function new_gen_nongaussian_training_data(
    state::StateMatrix;
    n::Integer=4096, batch_size=64, c=0.9, θ_range=(0., 2π), x_range=(-10., 10.),
    show_log=true
)
    data = Matrix{Float64}(undef, n, 2)
    p = (θ, x) -> SqState.pdf(state, θ, x)

    show_log && @info "Initial g"
    kde_result = kde((rand2range(rand(n),θ_range), rand2range(rand(n), x_range)))
    g = (θ, x) -> KernelDensity.pdf(kde_result, θ, x)

	sp_lock = Threads.SpinLock()
    Threads.@threads for i in 1:batch_size
        new_data = Vector{Float64}(undef, 2)
        accept_reject!(new_data, p, g, c, θ_range, x_range)

        lock(sp_lock) do
            view(data, i, :) .= new_data
        end
    end

	kde_result = kde((data[1:batch_size, 1], data[1:batch_size, 2]))
	g = (θ, x) -> KernelDensity.pdf(kde_result, θ, x)


    show_log && @info "Start to generate data"
    batch = div(n, batch_size)
    for i in 2:batch
        gen_batch_nongaussian_training_data!(
			data, 1:(i-1)*batch_size, (i-1)*batch_size.+(1:batch_size),
			p, g, c, θ_range, x_range
		)
        kde_result = kde((data[1:i*batch_size, 1], data[1:i*batch_size, 2]))
        g = (θ, x) -> KernelDensity.pdf(kde_result, θ, x)
        show_log && @info "progress: $i/$batch"
    end

    return data, kde_result
end

# ╔═╡ 5fd59cd2-0e5a-4905-887c-42672be81f76
state = displace!(
	squeeze!(
		SinglePhotonState(rep=StateMatrix, dim=100),
		ξ(0.5, π/2)
	),
	α(3., π/2)
)

# ╔═╡ c2980a70-c4a3-48fe-99de-f49300d85597
@time data, _ = new_gen_nongaussian_training_data(state)

# ╔═╡ c50f8825-10b7-447d-875f-31dea7025e5d
scatter(data[:, 1], data[:, 2], xlim=(0, 2π), ylim=(-10, 10), legend=false, size=(800, 400))

# ╔═╡ 0b3c1b51-8fd0-4f9b-b9d2-9037538e4dff
p = SqState.pdf(state, 0:0.1:2π, -10:0.1:10);

# ╔═╡ b9cc9ebe-f0fa-460d-b67b-a22dfc603de6
heatmap(
	p',
	ticks=[],
	color=:coolwarm,
	clim=(-1, 1),
	size=(800, 400),
	title="Probability density function"
)

# ╔═╡ Cell order:
# ╠═bb7cc2c4-db59-11eb-09a5-fbde282f71e6
# ╠═b01fbeb1-ad18-4788-becb-c97c6421038f
# ╠═0c700e42-99d5-4308-b63a-966c332fca6a
# ╠═22b8d28a-7beb-4af3-83a8-482bed72bead
# ╠═54d609b8-58f0-4f1e-aa1b-1ac6319184b9
# ╠═5fd59cd2-0e5a-4905-887c-42672be81f76
# ╠═c2980a70-c4a3-48fe-99de-f49300d85597
# ╠═c50f8825-10b7-447d-875f-31dea7025e5d
# ╠═0b3c1b51-8fd0-4f9b-b9d2-9037538e4dff
# ╠═b9cc9ebe-f0fa-460d-b67b-a22dfc603de6
