### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 35c2ae32-d1ba-11eb-0530-2b0f40549d44
begin
	using SqState
	using DataDeps
	using JLD2
	using Plots
	gr()
end

# ╔═╡ c1f6f093-c12b-484f-9f4f-73b978b4130c
begin
	data_path = joinpath(datadep"SqState", "training_data")
	readdir(data_path)
end

# ╔═╡ fad3101d-46b4-4089-89ab-b40c73315069
f = jldopen(joinpath(data_path, "17530388844665.jld2"), "r")

# ╔═╡ 5212b01a-3446-4f77-bc45-9585752bda65
begin
	wf = WignerFunction(-10:0.1:10, -10:0.1:10, dim=100)
	
	l = @layout [p;p]
	
	to_f5(x) = round(x, digits=5)
	
	function snap()
		i=rand(1:f["n_data"])
		r, θ, n̄, bias_phase = f["args"][:, i]
		title="r=$(to_f5(r)), θ=$(to_f5(θ)), n̄=$(to_f5(n̄)), dϕ=$(to_f5(bias_phase)))"

		points_plot = scatter(
			f["points"][:, i], 
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

# ╔═╡ Cell order:
# ╠═35c2ae32-d1ba-11eb-0530-2b0f40549d44
# ╠═c1f6f093-c12b-484f-9f4f-73b978b4130c
# ╠═fad3101d-46b4-4089-89ab-b40c73315069
# ╠═5212b01a-3446-4f77-bc45-9585752bda65
# ╠═d08178ec-2f1c-41af-8a74-0f8160f35dbe
# ╟─afc535e8-f188-48e0-8c6e-bc8eb6609e74
# ╟─ddc6077f-8f2c-4837-b8b8-137c81bf4456
