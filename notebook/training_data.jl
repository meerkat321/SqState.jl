### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 35c2ae32-d1ba-11eb-0530-2b0f40549d44
begin
	using BenchmarkTools
	using SqState
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

# ╔═╡ 2fa2ca29-e6b0-472e-b1aa-ad6fe0d77a88
md"
## Benchmark of data generator
"

# ╔═╡ 5ba6d286-6d56-42cf-ae5b-4746587eb07a
md"
### Gaussian state

Squeezed thermal state: $\hat{S}(\xi) \rho_{th}$

ξ = 0.3 exp(iπ/8)

n̄ = 0.5
"

# ╔═╡ 48ca97fa-e8c9-4e85-938b-a97bdbe5bf63
gaussian_state = SqueezedThermalState(ξ(0.3, π/8), 0.5, dim=100);

# ╔═╡ a1e155c8-9cc6-4828-8f69-73202ad9fff3
begin
	gaussian_points = Vector{Float64}(undef, 4096)
	@benchmark SqState.gen_gaussian_training_data!(
		gaussian_points,
		gaussian_state,
		0.
	)
end

# ╔═╡ b0ca2d64-b75c-4925-9a1f-5cdf5764a606
md"
**To generate 500k data: about 11(hr)**
"

# ╔═╡ 510d01ca-0394-4db9-982d-21a361132b69
md"
#### Take a glance of data
"

# ╔═╡ c1f6f093-c12b-484f-9f4f-73b978b4130c
begin
	data_path = joinpath(datadep"SqState", "training_data")
	readdir(data_path)
end

# ╔═╡ fad3101d-46b4-4089-89ab-b40c73315069
f = jldopen(joinpath(data_path, "10450874168442.jld2"), "r")

# ╔═╡ 5212b01a-3446-4f77-bc45-9585752bda65
begin
	wf = WignerFunction(-10:0.1:10, -10:0.1:10, dim=100)
	to_f5(x) = round(x, digits=5)

	function snap(; i=rand(1:f["n_data"]))
		r, θ, n̄, bias_phase = f["args"][:, i]
		title="r=$(to_f5(r)), θ=$(to_f5(θ)), n̄=$(to_f5(n̄)), dϕ=$(to_f5(bias_phase)))"

		points_plot = scatter(
			f["points"][:, i],
			ticks=[],
			title=title,
			legend=false,
			size=(800, 400)
		)
		w_plot = plot_wigner(
			wf(SqueezedThermalState(ξ(r, θ), n̄, dim=100)),
			SqState.Contour
		)

		return points_plot, w_plot
	end
end

# ╔═╡ d08178ec-2f1c-41af-8a74-0f8160f35dbe
d, w = snap();

# ╔═╡ afc535e8-f188-48e0-8c6e-bc8eb6609e74
d

# ╔═╡ ddc6077f-8f2c-4837-b8b8-137c81bf4456
w

# ╔═╡ b3602a8e-8e90-4e48-9f20-7ffa32e38807
md"
### Non-Gaussian State

Coherent squeezed single photon state: $\hat{D}(\alpha)\hat{S}(\xi)|1\rangle$

ξ = 0.5 exp(iπ/2)

α = 3.0 exp(iπ/2)
"

# ╔═╡ be6c7a83-1928-4467-b702-e2be1aaa0a75
non_gaussian_state = displace!(
	squeeze!(
		SinglePhotonState(rep=StateMatrix, dim=100),
		ξ(0.5, π/2)
	),
	α(3., π/2)
);

# ╔═╡ 49f54db7-71c1-40cd-a5bb-b38863457939
plot_wigner(wf(non_gaussian_state), SqState.Contour)

# ╔═╡ e73c6bb3-d590-47d0-9b40-d489664846f1
non_gaussuan_data = gen_nongaussian_training_data(non_gaussian_state);

# ╔═╡ 3c778dd6-8480-47ad-8672-450d29f4a274
scatter(
	non_gaussuan_data[1, :],
	non_gaussuan_data[2, :],
	ticks=[],
	legend=false,
	size=(800, 400),
	title="Non-Gaussian data"
)

# ╔═╡ 0d1a4641-dd7f-46ea-8138-46ff67e3a047
p = pdf(non_gaussian_state, 0:0.1:2π, -10:0.1:10);

# ╔═╡ 28a5a074-cf16-4c77-966b-032b7f7ca90e
heatmap(
	p',
	ticks=[],
	color=:coolwarm,
	clim=(-1, 1),
	size=(800, 400),
	title="Probability density function"
)

# ╔═╡ b7610707-aaeb-4ab2-92a6-d5e44cb6d90b
begin
	sampled_points = Matrix{Float64}(undef, 2, 4096)
    𝛑̂_res_vec = [
		Matrix{complex(Float64)}(
			undef, 
			non_gaussian_state.dim, 
			non_gaussian_state.dim
		) 
		for _ in 1:Threads.nthreads()
	]
	@benchmark SqState.gen_nongaussian_training_data!(
		sampled_points, 𝛑̂_res_vec, 
		non_gaussian_state,
		128, 64, 0.9, (0., 2π), (-10, 10),
		false
	)
end

# ╔═╡ Cell order:
# ╟─a9f16021-8559-47e8-a807-4a72e7940093
# ╟─35c2ae32-d1ba-11eb-0530-2b0f40549d44
# ╟─2fa2ca29-e6b0-472e-b1aa-ad6fe0d77a88
# ╟─5ba6d286-6d56-42cf-ae5b-4746587eb07a
# ╠═48ca97fa-e8c9-4e85-938b-a97bdbe5bf63
# ╟─a1e155c8-9cc6-4828-8f69-73202ad9fff3
# ╟─b0ca2d64-b75c-4925-9a1f-5cdf5764a606
# ╟─510d01ca-0394-4db9-982d-21a361132b69
# ╠═c1f6f093-c12b-484f-9f4f-73b978b4130c
# ╠═fad3101d-46b4-4089-89ab-b40c73315069
# ╟─5212b01a-3446-4f77-bc45-9585752bda65
# ╠═d08178ec-2f1c-41af-8a74-0f8160f35dbe
# ╟─afc535e8-f188-48e0-8c6e-bc8eb6609e74
# ╟─ddc6077f-8f2c-4837-b8b8-137c81bf4456
# ╟─b3602a8e-8e90-4e48-9f20-7ffa32e38807
# ╠═be6c7a83-1928-4467-b702-e2be1aaa0a75
# ╟─49f54db7-71c1-40cd-a5bb-b38863457939
# ╠═e73c6bb3-d590-47d0-9b40-d489664846f1
# ╟─3c778dd6-8480-47ad-8672-450d29f4a274
# ╟─0d1a4641-dd7f-46ea-8138-46ff67e3a047
# ╟─28a5a074-cf16-4c77-966b-032b7f7ca90e
# ╟─b7610707-aaeb-4ab2-92a6-d5e44cb6d90b
