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
end

# ╔═╡ c55d7af1-0cd3-4b0a-bbd5-0048f15ea851
m = SqState.get_model("model_q2ρ")

# ╔═╡ 8f82e42f-8c2c-4bd0-ab20-95b0930670dc
args = SqState.rand_arg((0, 1), (0, 2π), (0, 1))

# ╔═╡ f0cfd6f3-1e55-46cf-bd58-937e52cb7daa
begin
	dim = 100
	state = SqState.construct_state(args..., 1000)
	d = rand(state, 4096, IsGaussian)
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

# ╔═╡ Cell order:
# ╠═4655bdc2-1ed7-11ec-022d-7df0b8fdb907
# ╠═661cc8fe-44f3-422a-91f5-96956b748847
# ╠═c55d7af1-0cd3-4b0a-bbd5-0048f15ea851
# ╠═8f82e42f-8c2c-4bd0-ab20-95b0930670dc
# ╠═f0cfd6f3-1e55-46cf-bd58-937e52cb7daa
# ╠═08ddc8c4-57c7-4b2f-baeb-c424c3227065
# ╟─e9c31f3c-4f06-43b4-9982-4f69e997cde8
# ╟─6c0a3a23-335c-4661-8a2f-3729b3f0a1de
