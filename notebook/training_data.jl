### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ a5ac3616-158c-4d9d-a965-7bea0ac1283b
begin
	import Pkg
	Pkg.develop(path="/home/admin/Documents/GitHub/SqState.jl")
	Pkg.add("BenchmarkTools")
	Pkg.add("QuantumStateBase")
	Pkg.add("QuantumStatePlots")
	Pkg.add("DataDeps")
	Pkg.add("JLD2")
	Pkg.add("Plots")

	using SqState
	using BenchmarkTools
	using LinearAlgebra
	using QuantumStateBase
	using QuantumStatePlots
	using DataDeps
	using JLD2
	using Plots
	gr()
end

# ╔═╡ a9f16021-8559-47e8-a807-4a72e7940093
md"
# Training Data

JingYu
"

# ╔═╡ 510d01ca-0394-4db9-982d-21a361132b69
md"
#### Take a glance of data
"

# ╔═╡ c1f6f093-c12b-484f-9f4f-73b978b4130c
files = readdir(SqState.training_data_path())

# ╔═╡ fad3101d-46b4-4089-89ab-b40c73315069
f = jldopen(joinpath(SqState.training_data_path(), files[1]), "r")

# ╔═╡ 5212b01a-3446-4f77-bc45-9585752bda65
begin
	wf = WignerFunction(-10:0.1:10, -10:0.1:10, dim=100)
	to_f5(x) = round(x, digits=5)

	function snap(; i=rand(1:f["n_data"]))
		r, θ, n̄, bias_phase = f["args"][:, i]
		title="r=$(to_f5(r)), θ=$(to_f5(θ)), n̄=$(to_f5(n̄)), dϕ=$(to_f5(bias_phase)))"

		points_plot = scatter(
			f["points"][1, :, i],
			f["points"][2, :, i],
			# ticks=[],
			title=title,
			legend=false,
			size=(800, 400)
		)
		w_plot = plot_wigner(
			wf(SqueezedThermalState(ξ(r, θ), n̄, dim=100)),
			Contour
		)

		return points_plot, w_plot
	end
end

# ╔═╡ d08178ec-2f1c-41af-8a74-0f8160f35dbe
# d, w = snap();

# ╔═╡ afc535e8-f188-48e0-8c6e-bc8eb6609e74
# d

# ╔═╡ ddc6077f-8f2c-4837-b8b8-137c81bf4456
# w

# ╔═╡ c3552f01-8bfc-48c9-9c46-20b87114c810
m = get_model("model");

# ╔═╡ 9bbe4ab3-06a3-4dd0-8e05-acd977accfb8
begin
	new_state = SqueezedThermalState(ξ(1.34, 3.7), 0.08)
	new_data = rand(new_state, 4096, IsGaussian)
end

# ╔═╡ 1d5ed4d0-ce20-4a7a-9e62-2886e3b68889
begin
	a = 𝛒(new_state)+Matrix{Float64}(I, 70, 70)*1e-15
	# SqState.𝛒2y(a)
	# b = cholesky(Hermitian(a)).L
	# vcat([diag(b, i-70) for i in 1:70]...)

	# b*b'- Matrix{Float64}(I, 70, 70)*1e-15 ≈ a
end

# ╔═╡ a0e5e657-3c62-44f6-9a6d-20fdb69ce4fb
# scatter(new_data[1, :], new_data[2, :], size=(800, 400))

# ╔═╡ 398b7a48-ffcb-40f2-b2b3-92b8ce0a4354
begin
	l_new = reshape(m(reshape(new_data[2, :], (4096, 1, 1))), 4900)
end

# ╔═╡ e470b297-0ec0-48bb-ad75-309626e48fee
begin
	function merge_l(l_raw, dim)
		b = Int64((dim^2 - dim)/2 + dim)
		l = ComplexF64.(l_raw[1:b])
		for (i, e) in enumerate(l_raw[(b+1):end])
			l[i] += im * e
		end

		return l
	end

	function reshape_l(l, dim)
		l_ch = zeros(dim, dim)
		start_i = 1
		for i in -(dim-1):0
			l_ch += diagm(i => l[start_i:(start_i+(dim-1)+i)])
			start_i += (dim)+i
		end

		return l_ch
	end

	function ch2𝛒(l_ch, dim, δ)
		𝛒 = (l_ch' * l_ch) - Matrix{Float64}(I, dim, dim) * δ

		return 𝛒
	end

	function post_processing(l_raw; dim=70, δ=1e-15)
		return ch2𝛒(reshape_l(merge_l(l_raw, dim), dim), dim, δ)
	end

	𝛒_new = post_processing(l_new)

	# merge_l(l_new, 70)
	# reshape_l(merge_l(l_new, 70), 70)
end

# ╔═╡ 9c15467b-82c0-4e73-9e39-789c0b3ba45f
# sum(𝛒_new - 𝛒(new_state))

# ╔═╡ 9a9d0f4f-a61e-4dbb-b545-a711a3110e6e
plot_wigner(
	WignerFunction(-10:0.1:10, -10:0.1:10, dim=70)(StateMatrix(𝛒_new, 70)),
	Contour
)

# ╔═╡ Cell order:
# ╟─a9f16021-8559-47e8-a807-4a72e7940093
# ╠═a5ac3616-158c-4d9d-a965-7bea0ac1283b
# ╟─510d01ca-0394-4db9-982d-21a361132b69
# ╠═c1f6f093-c12b-484f-9f4f-73b978b4130c
# ╠═fad3101d-46b4-4089-89ab-b40c73315069
# ╟─5212b01a-3446-4f77-bc45-9585752bda65
# ╠═d08178ec-2f1c-41af-8a74-0f8160f35dbe
# ╠═afc535e8-f188-48e0-8c6e-bc8eb6609e74
# ╠═ddc6077f-8f2c-4837-b8b8-137c81bf4456
# ╠═c3552f01-8bfc-48c9-9c46-20b87114c810
# ╠═9bbe4ab3-06a3-4dd0-8e05-acd977accfb8
# ╠═1d5ed4d0-ce20-4a7a-9e62-2886e3b68889
# ╠═a0e5e657-3c62-44f6-9a6d-20fdb69ce4fb
# ╠═398b7a48-ffcb-40f2-b2b3-92b8ce0a4354
# ╠═e470b297-0ec0-48bb-ad75-309626e48fee
# ╠═9c15467b-82c0-4e73-9e39-789c0b3ba45f
# ╠═9a9d0f4f-a61e-4dbb-b545-a711a3110e6e
