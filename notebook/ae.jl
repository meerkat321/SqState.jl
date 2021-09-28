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
m = SqState.get_model("model_q2ρ1_70")

# ╔═╡ 8f82e42f-8c2c-4bd0-ab20-95b0930670dc
# args = SqState.rand_arg((0, 1), (0, 2π), (0, 1))
args = (0.9,  4.17646, 0.5, 0.3, 0.852858, 0.147142);

# ╔═╡ f0cfd6f3-1e55-46cf-bd58-937e52cb7daa
begin
	dim = 70
	state = SqState.construct_state(args..., 1000)
	d = Float32.(rand(state, 4096, IsGaussian))
	ρ = m(reshape(d[1, :], :, 1, 1))
	ρ = reshape(ρ[:, 1, 1] + im * ρ[:, 2, 1], dim, dim)
end;

# ╔═╡ 08ddc8c4-57c7-4b2f-baeb-c424c3227065
show_dim = 35

# ╔═╡ e9c31f3c-4f06-43b4-9982-4f69e997cde8
begin
	m1 = maximum(real.(state.𝛒))
	heatmap(real.(state.𝛒)[1:show_dim, 1:show_dim], color=:coolwarm, clim=(-m1, m1))
end

# ╔═╡ 6c0a3a23-335c-4661-8a2f-3729b3f0a1de
begin
	m2 = maximum(real.(ρ))
	heatmap(real.(ρ)[1:show_dim, 1:show_dim], color=:coolwarm, clim=(-m2, m2))
end

# ╔═╡ 6a68f473-cb16-4635-a3b0-9aae0e18d86d
function fidelity(ρ1, ρ2)
	sqrt_ρ1 = sqrt(ρ1)
	return tr(sqrt(sqrt_ρ1*ρ2*sqrt_ρ1))^2
end

# ╔═╡ 00f15b97-52e0-43aa-a2ee-136c7a8a6e95
fidelity(ρ, state.𝛒[1:dim, 1:dim])

# ╔═╡ 35a5ee23-a2df-47c6-a096-a5e1c60328b6
begin
	f_dim = 70
	
	f = 0
	for _ in 1:50
		f_state = SqState.construct_state(
			SqState.rand_arg((0, 1), (0, 2π), (0, 1))..., 
			1000
		)
		f_ρ = f_state.𝛒[1:f_dim, 1:f_dim]

		f_ρ̂ = m(reshape(Float32.(rand(state, 4096, IsGaussian))[1, :], :, 1, 1))
		f_ρ̂ = reshape(f_ρ̂[:, 1, 1] + im * f_ρ̂[:, 2, 1], f_dim, f_dim)
		f += fidelity(f_ρ̂, f_ρ)
	end
end

# ╔═╡ 0171f5c1-2b14-4694-b8b0-14e50591db4b
f/50

# ╔═╡ Cell order:
# ╠═4655bdc2-1ed7-11ec-022d-7df0b8fdb907
# ╠═661cc8fe-44f3-422a-91f5-96956b748847
# ╠═c55d7af1-0cd3-4b0a-bbd5-0048f15ea851
# ╠═8f82e42f-8c2c-4bd0-ab20-95b0930670dc
# ╠═f0cfd6f3-1e55-46cf-bd58-937e52cb7daa
# ╠═08ddc8c4-57c7-4b2f-baeb-c424c3227065
# ╟─e9c31f3c-4f06-43b4-9982-4f69e997cde8
# ╟─6c0a3a23-335c-4661-8a2f-3729b3f0a1de
# ╟─6a68f473-cb16-4635-a3b0-9aae0e18d86d
# ╠═00f15b97-52e0-43aa-a2ee-136c7a8a6e95
# ╠═35a5ee23-a2df-47c6-a096-a5e1c60328b6
# ╟─0171f5c1-2b14-4694-b8b0-14e50591db4b
