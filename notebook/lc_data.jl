### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 6e7b9a60-0cda-11ec-231a-39c70a5fb13c
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ a58a1153-502d-41d6-b69f-d65302e7e744
begin
	using QuantumStateBase
	using Plots
end

# ╔═╡ 424809ae-4e45-4da0-8513-ca3062b6118f
r, θ, n̄, c1, c2, c3 = 1., π/2, 1., 0.5, 0.4, 0.1

# ╔═╡ fd83678d-fbef-43da-a826-8d55f7111c27
point_dim = 1000

# ╔═╡ 0e5be1f7-caf8-4c3f-87ef-b7e028326ede
state =
	c1 * SqueezedState(ξ(r, θ), dim=point_dim, rep=StateMatrix) +
	c2 * SqueezedThermalState(ξ(r, θ), n̄, dim=point_dim) +
	c3 * ThermalState(n̄, dim=point_dim);

# ╔═╡ ca4f0095-132f-4154-a67c-e36c87ca3607
data = rand(state, 4096, IsGaussian)

# ╔═╡ d8f07973-cb35-4730-bbcb-c3a760436f4e
scatter(data[1, :], data[2, :], ylim=(-5, 5))

# ╔═╡ 5a1c07d7-ac58-4b6b-bc32-0a702e6f73d2
begin
	s = rand(SqueezedThermalState(ξ(2.0, 1π), 1., dim=1000), 4096, IsGaussian)
end

# ╔═╡ 926c5f29-075f-41d6-af94-6373909cbc9c
scatter(s[1, :], s[2, :], ylim=(-20, 20))

# ╔═╡ Cell order:
# ╠═6e7b9a60-0cda-11ec-231a-39c70a5fb13c
# ╠═a58a1153-502d-41d6-b69f-d65302e7e744
# ╠═424809ae-4e45-4da0-8513-ca3062b6118f
# ╠═fd83678d-fbef-43da-a826-8d55f7111c27
# ╠═0e5be1f7-caf8-4c3f-87ef-b7e028326ede
# ╠═ca4f0095-132f-4154-a67c-e36c87ca3607
# ╠═d8f07973-cb35-4730-bbcb-c3a760436f4e
# ╠═5a1c07d7-ac58-4b6b-bc32-0a702e6f73d2
# ╠═926c5f29-075f-41d6-af94-6373909cbc9c
