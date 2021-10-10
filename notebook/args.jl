### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 4655bdc2-1ed7-11ec-022d-7df0b8fdb907
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 661cc8fe-44f3-422a-91f5-96956b748847
begin
	using SqState
	using Plots
	using QuantumStateBase
	using Flux
	using LinearAlgebra
end

# ╔═╡ c55d7af1-0cd3-4b0a-bbd5-0048f15ea851
m = SqState.get_model("model_args_sqth_th")

# ╔═╡ 8f82e42f-8c2c-4bd0-ab20-95b0930670dc
# args = (0.9,  4.17646, 0.4);
# args = SqState.rand_arg_sqth_th((0, 1), (0, 2π), (0, 1))
args = (1., π/3, 0.4, 0.07, 0.8, 0.2)

# ╔═╡ f0cfd6f3-1e55-46cf-bd58-937e52cb7daa
begin
	dim = 100
	state = SqState.construct_state_sqth_th(args..., 1000)
	d = Float32.(rand(state, 4096, IsGaussian))
	argŝ = m(reshape(d[2, :], :, 1, 1))
	statê = SqState.construct_state_sqth_th(argŝ..., 1000)
	ρ = statê.𝛒
	argŝ
end

# ╔═╡ 57e75729-4bf4-4e0c-9ef2-bdaf6fb4e22b
function fidelity(ρ1, ρ2)
	sqrt_ρ1 = sqrt(ρ1)
	return tr(sqrt(sqrt_ρ1*ρ2*sqrt_ρ1))^2
end

# ╔═╡ 6739ac26-7590-46a8-a75a-2160a1d7c11d
fidelity(ρ, state.𝛒)

# ╔═╡ 00c09086-1eef-4b3f-b764-fe7201e1a354
scatter(d[1, :], d[2, :])

# ╔═╡ be08410c-4fce-405c-9b27-ababf152402e
begin
	d̂ = rand(statê, 4096, IsGaussian)
	scatter(d̂[1, :], d̂[2, :])
end

# ╔═╡ 32c40e60-0989-44f6-b518-9f6203aa6209


# ╔═╡ Cell order:
# ╠═4655bdc2-1ed7-11ec-022d-7df0b8fdb907
# ╠═661cc8fe-44f3-422a-91f5-96956b748847
# ╠═c55d7af1-0cd3-4b0a-bbd5-0048f15ea851
# ╠═8f82e42f-8c2c-4bd0-ab20-95b0930670dc
# ╠═f0cfd6f3-1e55-46cf-bd58-937e52cb7daa
# ╟─57e75729-4bf4-4e0c-9ef2-bdaf6fb4e22b
# ╟─6739ac26-7590-46a8-a75a-2160a1d7c11d
# ╠═00c09086-1eef-4b3f-b764-fe7201e1a354
# ╠═be08410c-4fce-405c-9b27-ababf152402e
# ╠═32c40e60-0989-44f6-b518-9f6203aa6209
