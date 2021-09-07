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

# ╔═╡ 19754c31-09f5-4568-b54e-a4528addb515
dim = 200;

# ╔═╡ d7be84cb-4b60-4b7d-b6ea-b928a5a6f317
# begin
# 	f = matopen("all.mat", "w")
# 	write(f, "flow_argv", flow_argv)
# 	write(f, "prev_paper_argv", prev_paper_argv)
# 	write(f, "flow_dm", flow_𝛒)
# 	write(f, "prev_paper_dm", prev_paper_𝛒)
# 	write(f, "flow_w", flow_𝐰)
# 	write(f, "prev_paper_w", prev_paper_𝐰)
# 	close(f)
# end

# ╔═╡ 236e0f31-43e0-4339-b230-c2a72a55c13c
@bind i Slider(1:21)

# ╔═╡ 9a3fae9a-c8d4-42c1-92c0-90fdb1161397
@bind refresh Button("refresh!")

# ╔═╡ 5c46c1ad-dacb-4edf-9f1d-dd5861b660ae
refresh; m = get_model("model");

# ╔═╡ 634f65c4-1d48-4403-b5ba-82e85257519e
begin 
	refresh
	dir = "Flow"
	data_path = joinpath(SqState.data_path(), dir)
	files = filter(x->contains(x, "SQ"), readdir(data_path))#[end:-1:1]
	flow_argv = Matrix{Float64}(undef, 6, length(files))
	for i in 1:size(flow_argv, 2)
		dataᵢ = SqState.get_data(joinpath(dir, files[i]))
		flow_argv[:, i:i] .= SqState.infer_arg(dataᵢ, 10)
	end
	
	dir = "1stPaperData"
	data_path = joinpath(SqState.data_path(), dir)
	files = filter(x->contains(x, "SQ"), readdir(data_path))[end:-1:1]
	prev_paper_argv = Matrix{Float64}(undef, 6, length(files))
	for i in 1:size(prev_paper_argv, 2)
		dataᵢ = SqState.get_data(joinpath(dir, files[i]))
		prev_paper_argv[:, i:i] .= SqState.infer_arg(dataᵢ, 10)
	end
end

# ╔═╡ f93a3a99-c8ab-4554-b9b9-2e8ee009a6a9
begin
	flow_𝛒 = Array{ComplexF64}(undef, dim, dim, size(flow_argv, 2))
	for i in 1:size(flow_argv, 2)
		r, θ, n̄, c1, c2, c3 = flow_argv[:, i]
		c1, c2, c3 = c1/sum([c1, c2, c3]), c2/sum([c1, c2, c3]), c3/sum([c1, c2, c3])
		θ = 0.
		flow_𝛒[:, :, i] .= 𝛒(
			c1 * SqueezedState(ξ(r, θ), rep=StateMatrix, dim=dim) + 
			c2 * SqueezedThermalState(ξ(r, θ), n̄, dim=dim) + 
			c3 * ThermalState(n̄, dim=dim)
		)
	end
	
	prev_paper_𝛒 = Array{ComplexF64}(undef, dim, dim, size(prev_paper_argv, 2))
	for i in 1:size(prev_paper_argv, 2)
		r, θ, n̄, c1, c2, c3 = prev_paper_argv[:, i]
		c1, c2, c3 = c1/sum([c1, c2, c3]), c2/sum([c1, c2, c3]), c3/sum([c1, c2, c3])
		prev_paper_𝛒[:, :, i] .= 𝛒(
			c1 * SqueezedState(ξ(r, θ), rep=StateMatrix, dim=dim) + 
			c2 * SqueezedThermalState(ξ(r, θ), n̄, dim=dim) + 
			c3 * ThermalState(n̄, dim=dim)
		)
	end
end

# ╔═╡ dc2d3311-4b32-4d23-bddd-b7899d7e3511
begin
	wf = WignerFunction(LinRange(-10, 10, 101), LinRange(-10, 10, 101), dim=dim)
	prev_paper_𝐰 = Array{Float64}(undef, 101, 101, size(prev_paper_𝛒, 3))
	for i in size(prev_paper_𝛒, 3)
		prev_paper_𝐰[:, :, i] .= wf(StateMatrix(prev_paper_𝛒[:, :, 1], dim)).𝐰_surface
	end
	
	wf = WignerFunction(LinRange(-3, 3, 101), LinRange(-3, 3, 101), dim=dim)
	flow_𝐰 = Array{Float64}(undef, 101, 101, size(flow_𝛒, 3))
	for i in size(flow_𝛒, 3)
		flow_𝐰[:, :, i] .= wf(StateMatrix(flow_𝛒[:, :, 1], dim)).𝐰_surface
	end
end

# ╔═╡ 61f9eb55-4916-4666-89d3-53f9dddcacb3
plot_wigner(wf(StateMatrix(flow_𝛒[:, :, i], dim)), Contour)

# ╔═╡ d6b78977-76d8-457f-8750-ecbfda7bad13
scatter(flow_argv[1, :], size=(800, 200), legend=false, title="r")

# ╔═╡ c04c95e6-ea7f-41d5-967f-5b51cf34be6d
scatter(flow_argv[2, :], size=(800, 200), legend=false, title="θ")

# ╔═╡ e1b9bbb3-ac8d-42bc-a98e-3ab0c8b3725b
scatter(flow_argv[3, :], size=(800, 200), legend=false, title="n̄")

# ╔═╡ 7a39bb3e-5f9e-45a5-a7b8-30142844ebe9
begin
	plot(size=(800, 200), legend=:left)
	scatter!(flow_argv[4, :], label="c₁")
	scatter!(flow_argv[5, :], label="c₂")
	scatter!(flow_argv[6, :], label="c₃")
end

# ╔═╡ 70d83e1f-5e2c-4583-b2ae-7a7b09ce7da3
scatter(prev_paper_argv[1, :], size=(800, 200), legend=false, title="r")

# ╔═╡ 507a5e70-21ca-41e1-9218-8a280bef8b94
scatter(prev_paper_argv[2, :], size=(800, 200), legend=false, title="θ")

# ╔═╡ 3c765713-199b-4736-9554-4605cf65c6ea
scatter(prev_paper_argv[3, :], size=(800, 200), legend=false, title="n̄")

# ╔═╡ 601a8008-cc6a-4dd4-8170-4b4d74a3be15
begin
	plot(size=(800, 200), legend=:left)
	scatter!(prev_paper_argv[4, :], label="c₁")
	scatter!(prev_paper_argv[5, :], label="c₂")
	scatter!(prev_paper_argv[6, :], label="c₃")
end

# ╔═╡ Cell order:
# ╠═12aa895e-0dfb-11ec-0cf1-972537d99a98
# ╠═ad2ef451-25d1-4001-abf1-ee9c9314b0fe
# ╠═5c46c1ad-dacb-4edf-9f1d-dd5861b660ae
# ╠═634f65c4-1d48-4403-b5ba-82e85257519e
# ╠═19754c31-09f5-4568-b54e-a4528addb515
# ╠═f93a3a99-c8ab-4554-b9b9-2e8ee009a6a9
# ╠═dc2d3311-4b32-4d23-bddd-b7899d7e3511
# ╠═d7be84cb-4b60-4b7d-b6ea-b928a5a6f317
# ╠═61f9eb55-4916-4666-89d3-53f9dddcacb3
# ╟─236e0f31-43e0-4339-b230-c2a72a55c13c
# ╟─d6b78977-76d8-457f-8750-ecbfda7bad13
# ╟─c04c95e6-ea7f-41d5-967f-5b51cf34be6d
# ╟─e1b9bbb3-ac8d-42bc-a98e-3ab0c8b3725b
# ╟─7a39bb3e-5f9e-45a5-a7b8-30142844ebe9
# ╟─70d83e1f-5e2c-4583-b2ae-7a7b09ce7da3
# ╟─507a5e70-21ca-41e1-9218-8a280bef8b94
# ╟─3c765713-199b-4736-9554-4605cf65c6ea
# ╟─601a8008-cc6a-4dd4-8170-4b4d74a3be15
# ╟─9a3fae9a-c8d4-42c1-92c0-90fdb1161397
