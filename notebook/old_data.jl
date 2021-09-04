### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ b01c5094-0d94-11ec-0dad-d7bcdffa1366
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ cdbd99ce-f2be-420e-850b-bccbfabaef24
begin
	using SqState
	using QuantumStateBase
	using QuantumStatePlots
	using DataDeps
	using Plots
	using MAT
end;

# ╔═╡ e125b5fd-79a9-4080-9d09-ef051aacdc66
m = get_model("model")

# ╔═╡ e0856114-a0bd-4dcd-bf92-eb4a4c15713f
begin
	data_path = joinpath(SqState.data_path(), "1stPaperData")
	files = filter(x->contains(x, "SQ"), readdir(data_path))[end:-1:1]
end

# ╔═╡ e7153f4f-14cd-4186-b492-24a3003a8626
begin
	# dtxsq01.mat, dtxsq10.mat
	# f = matopen(joinpath(SqState.data_path(), "1stPaperData/dtxsq01.mat"), "r")
	# read(f, "dd")
end

# ╔═╡ 7c349ead-86f4-4ed3-b06a-4e43b72ac07c
begin
	argv = Matrix{Float64}(undef, 6, length(files))
	for i in 1:size(argv, 2)
		dataᵢ = SqState.get_data(joinpath("1stPaperData", files[i]))
		argv[:, i:i] .= SqState.infer_arg(dataᵢ, 10)
	end
end

# ╔═╡ f8bf724b-b104-44b0-8a3e-c596c52f033b
scatter(argv[1, :], size=(800, 200), legend=false, title="r")

# ╔═╡ 081a05a1-2f07-440c-8b4c-c39ec500a452
scatter(argv[2, :], size=(800, 200), legend=false, title="θ")

# ╔═╡ 8958ae74-ce50-40d1-aac7-da228cace07a
scatter(argv[3, :], size=(800, 200), legend=false, title="n̄")

# ╔═╡ 0316f8f5-c96e-4894-884b-748f6bfae3bb
begin
	plot(size=(800, 200), legend=:left)
	scatter!(argv[4, :], label="c₁")
	scatter!(argv[5, :], label="c₂")
	scatter!(argv[6, :], label="c₃")
end

# ╔═╡ 409cb567-1a55-4ba7-9026-f49d50dd9861
argv

# ╔═╡ aba7ae96-7fe4-4ed2-ae34-997da846367e
begin
	𝛒s = Array{ComplexF64}(undef, 35, 35, 9)
	for (i, f) in enumerate(files)
		r, θ, n̄, c₁, c₂, c₃ = SqState.infer_arg(
			SqState.get_data(joinpath("1stPaperData", f)), 
			100
		)
		state = 
			c₁ * SqueezedState(ξ(r, θ), rep=StateMatrix, dim=35) + 
			c₂ * SqueezedThermalState(ξ(r, θ), n̄, dim=35) + 
			c₃ * ThermalState(n̄, dim=35)
		𝛒s[:, :, i] .= 𝛒(state)
	end
end

# ╔═╡ 1d223334-9326-46cc-a6e0-9a61f6d4d21f
# begin
# 	dmfile = matopen(joinpath(SqState.data_path(), "1stPaperData/dm.mat"), "w")
# 	write(dmfile, "dm", 𝛒s); write(dmfile, "argv", argv)
# 	close(dmfile)
# end

# ╔═╡ Cell order:
# ╟─b01c5094-0d94-11ec-0dad-d7bcdffa1366
# ╠═cdbd99ce-f2be-420e-850b-bccbfabaef24
# ╠═e125b5fd-79a9-4080-9d09-ef051aacdc66
# ╠═e0856114-a0bd-4dcd-bf92-eb4a4c15713f
# ╠═e7153f4f-14cd-4186-b492-24a3003a8626
# ╠═7c349ead-86f4-4ed3-b06a-4e43b72ac07c
# ╟─f8bf724b-b104-44b0-8a3e-c596c52f033b
# ╟─081a05a1-2f07-440c-8b4c-c39ec500a452
# ╟─8958ae74-ce50-40d1-aac7-da228cace07a
# ╟─0316f8f5-c96e-4894-884b-748f6bfae3bb
# ╠═409cb567-1a55-4ba7-9026-f49d50dd9861
# ╠═aba7ae96-7fe4-4ed2-ae34-997da846367e
# ╠═1d223334-9326-46cc-a6e0-9a61f6d4d21f
