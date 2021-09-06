### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 12aa895e-0dfb-11ec-0cf1-972537d99a98
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ ad2ef451-25d1-4001-abf1-ee9c9314b0fe
begin
	using PlutoUI
	using SqState
	using QuantumStateBase
	using QuantumStatePlots
	using DataDeps
	using Plots
	using LinearAlgebra
	using MAT
end;

# ╔═╡ 6d192e93-b290-429e-8aa8-135e908a4b2a
[
	"model1"=>"hloss=0.03643, t=70", 
	"model1_adam"=>"hloss=0.23, t=32",
	"model1_decent"=>"hloss=0.065, t=61"
];

# ╔═╡ cd92e6bd-af62-4879-9f71-9a059754fbf5
begin
	dir = "1stPaperData"
	# dir = "Flow"
	data_path = joinpath(SqState.data_path(), dir)
	files = filter(x->contains(x, "SQ"), readdir(data_path))#[end:-1:1]
end

# ╔═╡ 19754c31-09f5-4568-b54e-a4528addb515
# dim = 100

# ╔═╡ 0d40169a-d3c9-4716-a011-e8039ccc5f09
# function infer_𝛒(data::Matrix, n_sample::Integer)
# 	𝛒 = Matrix{ComplexF64}(undef, dim, dim)
#     for _ in 1:n_sample
#         𝛒 += post_processing(
# 			reshape(m(reshape(Float32.(SqState.sample(data, 4096)), (:, 1, 1))), :),
# 			dim=dim
# 		)
#     end
# 	𝛒 ./= n_sample

#     return 𝛒
# end

# ╔═╡ 9e698140-f00f-456e-8e31-4e9de4586327
# begin
# 	𝛒s = Array{ComplexF64}(undef, dim, dim, 9)
# 	for (i, f) in enumerate(files)
# 		𝛒s[:, :, i] .= infer_𝛒(
# 			SqState.get_data(joinpath("1stPaperData", f)), 
# 			2
# 		)
# 	end
# end

# ╔═╡ 137d7d10-a9e9-4b2e-a3ad-c1f0095e62f9
# begin
# 	wf = WignerFunction(LinRange(-3, 3, 101), LinRange(-3, 3, 101), dim=dim)
# 	plot_wigner(wf(StateMatrix(𝛒s[:, :, i], dim)), Contour)
# end

# ╔═╡ 9a3fae9a-c8d4-42c1-92c0-90fdb1161397
@bind refresh Button("refresh!")

# ╔═╡ 5c46c1ad-dacb-4edf-9f1d-dd5861b660ae
refresh; m = get_model("model1_decent");

# ╔═╡ 634f65c4-1d48-4403-b5ba-82e85257519e
begin 
	refresh
	argv = Matrix{Float64}(undef, 6, length(files))
	for i in 1:size(argv, 2)
		dataᵢ = SqState.get_data(joinpath(dir, files[i]))
		argv[:, i:i] .= SqState.infer_arg(dataᵢ, 10)
	end
end

# ╔═╡ d6b78977-76d8-457f-8750-ecbfda7bad13
scatter(argv[1, :], size=(800, 200), legend=false, title="r")

# ╔═╡ c04c95e6-ea7f-41d5-967f-5b51cf34be6d
scatter(argv[2, :], size=(800, 200), legend=false, title="θ")

# ╔═╡ e1b9bbb3-ac8d-42bc-a98e-3ab0c8b3725b
scatter(argv[3, :], size=(800, 200), legend=false, title="n̄")

# ╔═╡ 7a39bb3e-5f9e-45a5-a7b8-30142844ebe9
begin
	plot(size=(800, 200), legend=:left)
	scatter!(argv[4, :], label="c₁")
	scatter!(argv[5, :], label="c₂")
	scatter!(argv[6, :], label="c₃")
end

# ╔═╡ ebcc83ca-25f9-4819-933b-dda3b1ec9381
@bind i Slider(1:9; default=1, show_value=true)

# ╔═╡ Cell order:
# ╠═12aa895e-0dfb-11ec-0cf1-972537d99a98
# ╠═6d192e93-b290-429e-8aa8-135e908a4b2a
# ╠═ad2ef451-25d1-4001-abf1-ee9c9314b0fe
# ╠═5c46c1ad-dacb-4edf-9f1d-dd5861b660ae
# ╠═cd92e6bd-af62-4879-9f71-9a059754fbf5
# ╟─19754c31-09f5-4568-b54e-a4528addb515
# ╟─0d40169a-d3c9-4716-a011-e8039ccc5f09
# ╟─9e698140-f00f-456e-8e31-4e9de4586327
# ╟─137d7d10-a9e9-4b2e-a3ad-c1f0095e62f9
# ╠═634f65c4-1d48-4403-b5ba-82e85257519e
# ╟─d6b78977-76d8-457f-8750-ecbfda7bad13
# ╟─c04c95e6-ea7f-41d5-967f-5b51cf34be6d
# ╟─e1b9bbb3-ac8d-42bc-a98e-3ab0c8b3725b
# ╟─7a39bb3e-5f9e-45a5-a7b8-30142844ebe9
# ╟─9a3fae9a-c8d4-42c1-92c0-90fdb1161397
# ╟─ebcc83ca-25f9-4819-933b-dda3b1ec9381
