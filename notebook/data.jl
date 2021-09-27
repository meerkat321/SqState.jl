### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ c65cb242-1f99-11ec-3f99-4fcf53839d59
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 15c5ae4b-1910-4728-ae27-c3f8a601746e
begin
	using SqState
	using Plots
	using QuantumStateBase
	using JLD2
end

# ╔═╡ f51d1bc5-5ffc-4f6d-a5b1-cee378907dd1
readdir(SqState.training_data_path())

# ╔═╡ f4c98102-6b12-4f04-b46a-d02ef0c72882
σs = jldopen(joinpath(SqState.training_data_path(), "sqth_th_2021-09-24T17_57_48.332.jld2"))["σs"]

# ╔═╡ c97e24d0-3c31-4624-8c88-6f182c0c161b
scatter(σs[:, 7])

# ╔═╡ Cell order:
# ╠═c65cb242-1f99-11ec-3f99-4fcf53839d59
# ╠═15c5ae4b-1910-4728-ae27-c3f8a601746e
# ╠═f51d1bc5-5ffc-4f6d-a5b1-cee378907dd1
# ╠═f4c98102-6b12-4f04-b46a-d02ef0c72882
# ╠═c97e24d0-3c31-4624-8c88-6f182c0c161b
