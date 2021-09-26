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
end

# ╔═╡ c55d7af1-0cd3-4b0a-bbd5-0048f15ea851
m = SqState.get_model("model_ae")

# ╔═╡ 8f82e42f-8c2c-4bd0-ab20-95b0930670dc
args = SqState.rand_arg((0, 2), (0, 2π), (0, 1))

# ╔═╡ d5452c99-db05-47eb-ac03-45c3d304c15a
state = SqState.construct_state(args..., 70);

# ╔═╡ f0cfd6f3-1e55-46cf-bd58-937e52cb7daa
d = rand(state, 4096, IsGaussian);

# ╔═╡ 4f9c8bb5-aac5-4d0a-927f-fe68e70bad9a
scatter(d[1, :], d[2, :])

# ╔═╡ 95473615-9b60-4379-a91f-6d0d7ce62577
σs = m(reshape(d[1, :], :, 1, 1));

# ╔═╡ 8135f4f8-64de-426d-b8c8-789d0bc12b5d
scatter(randn(4096) .* σs)

# ╔═╡ Cell order:
# ╠═4655bdc2-1ed7-11ec-022d-7df0b8fdb907
# ╠═661cc8fe-44f3-422a-91f5-96956b748847
# ╠═c55d7af1-0cd3-4b0a-bbd5-0048f15ea851
# ╠═8f82e42f-8c2c-4bd0-ab20-95b0930670dc
# ╟─d5452c99-db05-47eb-ac03-45c3d304c15a
# ╟─f0cfd6f3-1e55-46cf-bd58-937e52cb7daa
# ╟─4f9c8bb5-aac5-4d0a-927f-fe68e70bad9a
# ╠═95473615-9b60-4379-a91f-6d0d7ce62577
# ╠═8135f4f8-64de-426d-b8c8-789d0bc12b5d
