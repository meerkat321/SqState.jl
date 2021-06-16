### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 488178d4-ce64-11eb-1368-f55095c74c80
begin
	using SqState
	using JLD2
	using DataDeps
	using Plots
	plotly()
end

# ╔═╡ f1c41b26-6d57-44ef-8fe2-923439a7e3c4
begin
	data_path = joinpath(datadep"SqState", "training_data", "gen_data")
	file_names = readdir(data_path)
end

# ╔═╡ 48dd9d37-967f-4ed1-995b-4066c098e22d
f = jldopen(joinpath(data_path, "test.jld2"), "r")

# ╔═╡ a312a13e-d124-4fc7-bc8b-976fea4b4d80
f["bin_θs"]

# ╔═╡ 06a9e533-16a0-47c2-ab5b-e2305511dec7
f["bin_xs"]

# ╔═╡ 4fdbec65-fb84-475c-b889-ca38ce445060
f["dim"]

# ╔═╡ edec28c1-f697-402a-83d6-0a64dfe615fb
begin
	𝐩_dict = f["𝐩_dict"]
	args = hcat([[a...] for (a, _) in 𝐩_dict]...)
	𝐩s = [p for (_, p) in 𝐩_dict]
end;

# ╔═╡ 9b31f516-e9e9-43a4-b410-4e199c58efa4
tof5(f) = round(f, digits=3);

# ╔═╡ 282ee8c1-5362-4170-a8e8-2178a115e140
begin
	i = rand(1:length(𝐩s))
	r, θ, n̄ = args[:, i]
	𝐩 = 𝐩s[i]
	
	lim = maximum(abs.(𝐩))
	title = "r=$(tof5(r)) θ=$(tof5(θ)) n̄=$(tof5(n̄))"
	heatmap(𝐩', clim=(-lim, lim), title=title, color=:coolwarm)
end

# ╔═╡ Cell order:
# ╠═488178d4-ce64-11eb-1368-f55095c74c80
# ╠═f1c41b26-6d57-44ef-8fe2-923439a7e3c4
# ╠═48dd9d37-967f-4ed1-995b-4066c098e22d
# ╠═a312a13e-d124-4fc7-bc8b-976fea4b4d80
# ╠═06a9e533-16a0-47c2-ab5b-e2305511dec7
# ╠═4fdbec65-fb84-475c-b889-ca38ce445060
# ╠═edec28c1-f697-402a-83d6-0a64dfe615fb
# ╠═9b31f516-e9e9-43a4-b410-4e199c58efa4
# ╠═282ee8c1-5362-4170-a8e8-2178a115e140
