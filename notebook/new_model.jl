### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 12aa895e-0dfb-11ec-0cf1-972537d99a98
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ ad2ef451-25d1-4001-abf1-ee9c9314b0fe
begin
	using SqState
	using QuantumStateBase
	using QuantumStatePlots
	using DataDeps
	using Plots
	using LinearAlgebra
	using MAT
end;

# ╔═╡ 5c46c1ad-dacb-4edf-9f1d-dd5861b660ae
m = get_model("model1")

# ╔═╡ cd92e6bd-af62-4879-9f71-9a059754fbf5
begin
	data_path = joinpath(SqState.data_path(), "1stPaperData")
	files = filter(x->contains(x, "SQ"), readdir(data_path))[end:-1:1]
end

# ╔═╡ 0d40169a-d3c9-4716-a011-e8039ccc5f09
function infer_𝛒(data::Matrix, n_sample::Integer)
	𝛒 = Matrix{ComplexF64}(undef, 100, 100)
    for _ in 1:n_sample
        𝛒 += post_processing(
			reshape(m(reshape(Float32.(SqState.sample(data, 4096)), (:, 1, 1))), :)
		)
    end
	𝛒 ./= tr(𝛒)

    return 𝛒
end

# ╔═╡ 9e698140-f00f-456e-8e31-4e9de4586327
begin
	𝛒s = Array{ComplexF64}(undef, 100, 100, 9)
	for (i, f) in enumerate(files)
		𝛒s[:, :, i] .= infer_𝛒(
			SqState.get_data(joinpath("1stPaperData", f)), 
			10
		)
	end
end

# ╔═╡ 2671ab33-6f4c-4842-8739-9bd8669130ca
𝛒s

# ╔═╡ 137d7d10-a9e9-4b2e-a3ad-c1f0095e62f9
begin
	wf = WignerFunction(LinRange(-3, 3, 101), LinRange(-3, 3, 101), dim=100)
	plot_wigner(wf(StateMatrix(𝛒s[:, :, 9], 100)), Contour)
end

# ╔═╡ Cell order:
# ╠═12aa895e-0dfb-11ec-0cf1-972537d99a98
# ╠═ad2ef451-25d1-4001-abf1-ee9c9314b0fe
# ╠═5c46c1ad-dacb-4edf-9f1d-dd5861b660ae
# ╠═cd92e6bd-af62-4879-9f71-9a059754fbf5
# ╠═0d40169a-d3c9-4716-a011-e8039ccc5f09
# ╠═9e698140-f00f-456e-8e31-4e9de4586327
# ╠═2671ab33-6f4c-4842-8739-9bd8669130ca
# ╠═137d7d10-a9e9-4b2e-a3ad-c1f0095e62f9
