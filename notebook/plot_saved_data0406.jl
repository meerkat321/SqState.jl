### A Pluto.jl notebook ###
# v0.14.3

using Markdown
using InteractiveUtils

# ╔═╡ 7a59734e-a4d3-11eb-2592-298c3e22f145
begin
	using SqState
	using DataDeps
	using HDF5
	using Plots
	using DisplayAs
end

# ╔═╡ 4f82a067-edb5-43c3-b9f7-26532258d5d5
function plot_wigner_surface(file_name)
	w = h5open(datadep"SqState/data/data0406/WignerSurface.h5", "r") do file
        read(file, file_name)
	end

	max_z = maximum(abs.(w))

	xs = ps = -10:0.02:10
	surface(xs, ps, w, zlim=(-max_z, max_z),  color=:coolwarm, fillalpha=0.95) |> DisplayAs.PNG
end

# ╔═╡ 7d6957d7-86d9-4e75-8b1a-c496fc1af8d7
plot_wigner_surface("SQ3_model_output.h5")

# ╔═╡ Cell order:
# ╠═7a59734e-a4d3-11eb-2592-298c3e22f145
# ╠═4f82a067-edb5-43c3-b9f7-26532258d5d5
# ╠═7d6957d7-86d9-4e75-8b1a-c496fc1af8d7
