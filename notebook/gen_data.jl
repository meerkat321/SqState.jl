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
	𝐩_dict = jldopen(joinpath(data_path, file_names[1]), "r")["𝐩_dict"];
end;

# ╔═╡ 282ee8c1-5362-4170-a8e8-2178a115e140
begin
	plots = Vector{Plots.Plot}(undef, length(𝐩_dict))
	for (i, ((r, θ, n̄), 𝐩)) in enumerate(𝐩_dict)
		lim = maximum(abs.(𝐩))
		plots[i] = heatmap(𝐩', clim=(-lim, lim), color=:coolwarm)
	end
end

# ╔═╡ a58cd456-b960-44a0-9e29-81850711d6f3
begin
	i = rand(1:length(𝐩_dict))
	plot(plots[i])
end

# ╔═╡ Cell order:
# ╠═488178d4-ce64-11eb-1368-f55095c74c80
# ╠═f1c41b26-6d57-44ef-8fe2-923439a7e3c4
# ╠═282ee8c1-5362-4170-a8e8-2178a115e140
# ╠═a58cd456-b960-44a0-9e29-81850711d6f3
