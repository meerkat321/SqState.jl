### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 926e4270-ee52-11eb-15b9-0b59dac4c681
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 1d009b9d-dcf2-4bdf-8dd8-fef135cb8524
begin
	using SqState
	using QuantumStateBase
	using QuantumStatePlots
	using DataDeps
	using Plots
	using MAT
end;

# ╔═╡ 05865de0-1458-4c59-880c-8619d4c7dd83
md"
# Real time system

JingYu Ning

The real time quantum state tomogrophy system plays an important role in the homodyme experiment. It provides the features of data visualization without delay and get rid of the extremely complex data analasis works.

In this demonstration, we will show you how the data flows and how the machine learning model works.


"

# ╔═╡ d372a693-4a33-45b1-898c-72339f9e910f
md"
## A small yet elegant model

Apart from the intergration of the model implemented by HsenYi, the improvement of perforement is another essential part of real time system.

Therefore I implemented a small yet elegant CNN model that will infer the arguments `r`, `θ` and `n̄` for a squeezed thermal state to boost the inference process.
"

# ╔═╡ 7a22ff61-a32f-4def-9fd9-23a540791609
m = get_model("model")

# ╔═╡ 6f6ad264-cac8-4a04-906a-e83ebac85136
md"
## Experiment data from the lab

The experiments of the following data are accomplish by YiRu in NTHU
"

# ╔═╡ 1b4271a4-8540-4881-b6cb-7a3dd8e9d0b1
begin
	data_path = joinpath(SqState.data_path(), "Flow")
	files = readdir(data_path)
end

# ╔═╡ 22a0af68-9012-4cc1-aebf-2f91bac387b4
md"
## About the data

The data is generated by the homodyme detector and capture by the oscilloscope.
"

# ╔═╡ 55d14429-f6e6-4db3-a7ea-6aec92fccd7d
begin
	data = SqState.get_data(joinpath("Flow", files[end-1]))
	x_data = SqState.sample(data, 4096)
end

# ╔═╡ 49fe2f66-5f90-4ad3-8ab3-846b6684b9e7
scatter(LinRange(0, 2π, 4096), x_data, size=(800, 400), title="Quadrature data")

# ╔═╡ 8bb50bb4-dbae-4adb-acd6-46dc45d8c109
md"
## Let the model do the magic

the model will infere the arguments for the state via the given quaduture data.
"

# ╔═╡ b015f3e0-89d8-4ecd-a3e5-23d9cc83a6e5
r, θ, n̄, c1, c2, c3 = SqState.infer_arg(data, 100)

# ╔═╡ d02b6dbc-a45d-4900-881a-30b550564f6d
md"
After the inference, we can construct the quantum state and plot the Wigner function, or apply some operator to it.
"

# ╔═╡ 90a4e898-f95a-4bd1-8825-9c7d8561a683
state =
	c1 * SqueezedState(ξ(r, θ), rep=StateMatrix) +
	c2 * SqueezedThermalState(ξ(r, θ), n̄) +
	c3 * ThermalState(n̄)

# ╔═╡ d7756447-61ae-49f7-b0cb-ea1ff0432d29
begin
	wf = WignerFunction(LinRange(-3, 3, 101), LinRange(-3, 3, 101))
	plot_wigner(wf(state), Contour)
end

# ╔═╡ 8ff17755-21eb-45c2-adbc-27a5d021f79f
md"
Calculate all args for all data
"

# ╔═╡ 870e4dd3-f8e6-45a1-87ba-721237e00066
begin
	argv = Matrix{Float64}(undef, 6, length(files))
	for i in 1:size(argv, 2)
		dataᵢ = SqState.get_data(joinpath("Flow", files[i]))
		argv[:, i:i] .= SqState.infer_arg(dataᵢ, 10)
	end
end

# ╔═╡ 90e1807a-0779-4156-9b19-44cab50e2efc
scatter(argv[1, :], size=(800, 200), legend=false, title="r")

# ╔═╡ b24b71fc-2f37-408d-9888-318eeba180dd
scatter(argv[2, :], size=(800, 200), legend=false, title="θ")

# ╔═╡ c730f57d-1b41-4165-b50c-e564fbda8861
scatter(argv[3, :], size=(800, 200), legend=false, title="n̄")

# ╔═╡ 203ee767-c38b-4123-be29-0d237dc57c40
begin
	plot(size=(800, 200), legend=:left)
	scatter!(argv[4, :], label="c₁")
	scatter!(argv[5, :], label="c₂")
	scatter!(argv[6, :], label="c₃")
end

# ╔═╡ 5bb4f85c-af93-456c-af73-2dc069d0237a
md"
## Wigner flow
"

# ╔═╡ 3fdf7f29-b8b6-478b-9b4d-fb96407e99ae
begin
	function get_w(data_name::String; n_sample=10, fix_θ=true, dim=70)
		data = SqState.get_data(joinpath("Flow", data_name))
    	state, w = SqState.calc_w(
			SqState.infer_arg(data, n_sample)..., dim, fix_θ,
			wf=wf
		)

		return w
	end

	anim = @animate for f in files
		plot_wigner(get_w(f, fix_θ=true), QuantumStatePlots.Contour)
		annotate!(-2.5, 2.5, text("$f", :left))
	end

	gif(anim, fps=2)
end

# ╔═╡ c2fcff08-cbab-47f6-b5b3-5a2989b11c91
# begin
# 	𝐰s = Array{Float64}(undef, 101, 101, 21)
# 	for (i, f) in enumerate(files)
# 		𝐰s[:, :, i] .= get_w(f, fix_θ=true).𝐰_surface
# 	end

# 	wfile = matopen("w.mat", "w")
# 	write(wfile, "ws", 𝐰s)
# 	close(wfile)
# end

# ╔═╡ 87ca3223-89cf-4dc3-aa25-f66e9bb22a2b
# begin
# 	argvfile = matopen("argv.mat", "w")
# 	write(argvfile, "argv", argv)
# 	close(argvfile)
# end

# ╔═╡ e9ea5eba-70fc-40d0-8bb4-62a289305a7c
begin
	rs = [0.0952 0.0809 0.1271 0.1572 0.2652 0.2753 0.3454 0.3492]
	n̄s = [0.2549 0.2417 0.2283 0.2355 0.2402 0.2761 0.2793 0.2595]
	
	𝐰s_sqth = Array{Float64}(undef, 101, 101, 8)
	𝐰s_th = Array{Float64}(undef, 101, 101, 8)
	for (i, (r, n̄)) in enumerate(zip(rs, n̄s))
		𝐰s_sqth[:, :, i] .= wf(SqueezedThermalState(ξ(r, 0.), n̄)).𝐰_surface
		𝐰s_th[:, :, i] .= wf(ThermalState(n̄)).𝐰_surface
	end
	
	# w_sqth_th_file = matopen("w_sqth_th.mat", "w")
	# write(w_sqth_th_file, "w_sqth", 𝐰s_sqth)
	# write(w_sqth_th_file, "w_th", 𝐰s_th)
	# close(w_sqth_th_file)
end

# ╔═╡ e4ccfa2e-6988-40ff-8403-188998242e04
heatmap(𝐰s_sqth[:, :, 8])

# ╔═╡ Cell order:
# ╟─05865de0-1458-4c59-880c-8619d4c7dd83
# ╠═926e4270-ee52-11eb-15b9-0b59dac4c681
# ╠═1d009b9d-dcf2-4bdf-8dd8-fef135cb8524
# ╟─d372a693-4a33-45b1-898c-72339f9e910f
# ╠═7a22ff61-a32f-4def-9fd9-23a540791609
# ╟─6f6ad264-cac8-4a04-906a-e83ebac85136
# ╠═1b4271a4-8540-4881-b6cb-7a3dd8e9d0b1
# ╟─22a0af68-9012-4cc1-aebf-2f91bac387b4
# ╠═55d14429-f6e6-4db3-a7ea-6aec92fccd7d
# ╠═49fe2f66-5f90-4ad3-8ab3-846b6684b9e7
# ╟─8bb50bb4-dbae-4adb-acd6-46dc45d8c109
# ╠═b015f3e0-89d8-4ecd-a3e5-23d9cc83a6e5
# ╟─d02b6dbc-a45d-4900-881a-30b550564f6d
# ╠═90a4e898-f95a-4bd1-8825-9c7d8561a683
# ╠═d7756447-61ae-49f7-b0cb-ea1ff0432d29
# ╟─8ff17755-21eb-45c2-adbc-27a5d021f79f
# ╠═870e4dd3-f8e6-45a1-87ba-721237e00066
# ╟─90e1807a-0779-4156-9b19-44cab50e2efc
# ╟─b24b71fc-2f37-408d-9888-318eeba180dd
# ╟─c730f57d-1b41-4165-b50c-e564fbda8861
# ╟─203ee767-c38b-4123-be29-0d237dc57c40
# ╟─5bb4f85c-af93-456c-af73-2dc069d0237a
# ╠═3fdf7f29-b8b6-478b-9b4d-fb96407e99ae
# ╠═c2fcff08-cbab-47f6-b5b3-5a2989b11c91
# ╠═87ca3223-89cf-4dc3-aa25-f66e9bb22a2b
# ╠═e9ea5eba-70fc-40d0-8bb4-62a289305a7c
# ╠═e4ccfa2e-6988-40ff-8403-188998242e04
