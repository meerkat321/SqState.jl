### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ b62ddd88-06df-11ec-3646-9bb715679f3b
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ cdd017ab-acf0-4621-b4a4-fa3ede59e5ce
using QuantumStateBase, MAT

# ╔═╡ 37ff7147-e3e6-4bfa-b3e4-bc426294ec9d
wf = WignerFunction(LinRange(-5, 5, 100), LinRange(-5, 5, 100));

# ╔═╡ e122de1c-f4d3-469c-bcfd-f400805d88fa
function gen_w(r::Float64)
	state = SqueezedState(ξ(r, 0.))
	
	return wf(state).𝐰_surface
end

# ╔═╡ 6f240c6e-66fa-4de6-a4ab-e5072320ade7
rs = [0.085223, 0.0439472, 0.148323, 0.166576, 0.218475, 0.226697, 0.264381, 0.279566]

# ╔═╡ 6b68595c-7a86-453f-b330-3e08034934c4
ws = reshape(hcat(gen_w.(rs)...), 100, 100, :)

# ╔═╡ 4c43a789-4015-49d7-b3c8-851970679c85
begin
	file = matopen("w.mat", "w")
	write(file, "ws", ws)
	close(file)
end

# ╔═╡ 217bd716-b199-4fb8-9385-5961b2318c75


# ╔═╡ Cell order:
# ╠═b62ddd88-06df-11ec-3646-9bb715679f3b
# ╠═cdd017ab-acf0-4621-b4a4-fa3ede59e5ce
# ╠═37ff7147-e3e6-4bfa-b3e4-bc426294ec9d
# ╠═e122de1c-f4d3-469c-bcfd-f400805d88fa
# ╠═6f240c6e-66fa-4de6-a4ab-e5072320ade7
# ╠═6b68595c-7a86-453f-b330-3e08034934c4
# ╠═4c43a789-4015-49d7-b3c8-851970679c85
# ╠═217bd716-b199-4fb8-9385-5961b2318c75
