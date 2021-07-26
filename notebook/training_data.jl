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
## Take a glance of data
"

# ╔═╡ c1f6f093-c12b-484f-9f4f-73b978b4130c
begin
	files = readdir(SqState.training_data_path())
	f = jldopen(joinpath(SqState.training_data_path(), files[1]), "r")
	f["args"]
end

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
d, w = snap();

# ╔═╡ afc535e8-f188-48e0-8c6e-bc8eb6609e74
d

# ╔═╡ ddc6077f-8f2c-4837-b8b8-137c81bf4456
w

# ╔═╡ aa4b34dc-6ff6-470c-b011-df425e1ea638
md"
## Check model
"

# ╔═╡ c3552f01-8bfc-48c9-9c46-20b87114c810
m = get_model("model0")

# ╔═╡ 9bbe4ab3-06a3-4dd0-8e05-acd977accfb8
begin
	new_state = SqueezedThermalState(ξ(0.8, π/2), 0.3, dim=100)
	new_data = rand(new_state, 4096, IsGaussian)
end

# ╔═╡ a0e5e657-3c62-44f6-9a6d-20fdb69ce4fb
scatter(new_data[1, :], new_data[2, :], legend=false, size=(800, 400))

# ╔═╡ 398b7a48-ffcb-40f2-b2b3-92b8ce0a4354
r, θ, n̄ = m(reshape((new_data[2, :]), (4096, 1, 1)))

# ╔═╡ c792e32d-797c-4a96-9045-7ae3bd22d1ea
md"
**Theoretical**
"

# ╔═╡ df0c2738-5cbd-4265-9867-f4c5b2527461
plot_wigner(
	WignerFunction(-10:0.1:10, -10:0.1:10, dim=100)(new_state),
	Contour
)

# ╔═╡ d6b6dd28-30a6-4ae2-aaa9-fa1f49db079d
md"
**Model inference**
"

# ╔═╡ 9a9d0f4f-a61e-4dbb-b545-a711a3110e6e
plot_wigner(
	WignerFunction(-10:0.1:10, -10:0.1:10, dim=100)(
		SqueezedThermalState(ξ(r, θ), n̄, dim=100)
	),
	Contour
)

# ╔═╡ Cell order:
# ╟─a9f16021-8559-47e8-a807-4a72e7940093
# ╟─a5ac3616-158c-4d9d-a965-7bea0ac1283b
# ╟─510d01ca-0394-4db9-982d-21a361132b69
# ╠═c1f6f093-c12b-484f-9f4f-73b978b4130c
# ╟─5212b01a-3446-4f77-bc45-9585752bda65
# ╠═d08178ec-2f1c-41af-8a74-0f8160f35dbe
# ╠═afc535e8-f188-48e0-8c6e-bc8eb6609e74
# ╠═ddc6077f-8f2c-4837-b8b8-137c81bf4456
# ╟─aa4b34dc-6ff6-470c-b011-df425e1ea638
# ╠═c3552f01-8bfc-48c9-9c46-20b87114c810
# ╠═9bbe4ab3-06a3-4dd0-8e05-acd977accfb8
# ╠═a0e5e657-3c62-44f6-9a6d-20fdb69ce4fb
# ╠═398b7a48-ffcb-40f2-b2b3-92b8ce0a4354
# ╟─c792e32d-797c-4a96-9045-7ae3bd22d1ea
# ╠═df0c2738-5cbd-4265-9867-f4c5b2527461
# ╟─d6b6dd28-30a6-4ae2-aaa9-fa1f49db079d
# ╠═9a9d0f4f-a61e-4dbb-b545-a711a3110e6e
