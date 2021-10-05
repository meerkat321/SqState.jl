### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 4655bdc2-1ed7-11ec-022d-7df0b8fdb907
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 661cc8fe-44f3-422a-91f5-96956b748847
begin
	using SqState
	using Plots
	using QuantumStateBase
	using Flux
	using LinearAlgebra
end

# ╔═╡ 510b566f-4fd9-4d96-bfc5-b54385b713ce
using Fetch, DataDeps, MAT

# ╔═╡ c55d7af1-0cd3-4b0a-bbd5-0048f15ea851
# m = SqState.get_model("model_q2args_sqth_x")

# ╔═╡ 8f82e42f-8c2c-4bd0-ab20-95b0930670dc
# args = (1.72,  4.17646, 0.4);
args = SqState.rand_arg_sqth((0.23, 0.231), (0, 2π), (0, 0.3))
# args = (1., π/3, 0.4, 0.07, 0.8, 0.2)

# ╔═╡ 9588e66c-9c19-4257-8fd2-12455e0c0f0d
relu(x) = x<0 ? zero(typeof(x)) : x

# ╔═╡ f0cfd6f3-1e55-46cf-bd58-937e52cb7daa
begin
	dim = 100
	state = SqState.construct_state_sqth(args..., 1000)
	d = Float32.(rand(state, 4096, IsGaussian))
	argŝ = relu.(m(reshape(d, 2, :, 1)))
	statê = SqState.construct_state_sqth(argŝ..., 1000)
	ρ = statê.𝛒
	argŝ
end

# ╔═╡ 57e75729-4bf4-4e0c-9ef2-bdaf6fb4e22b
function fidelity(ρ1, ρ2)
	sqrt_ρ1 = sqrt(ρ1)
	return tr(sqrt(sqrt_ρ1*ρ2*sqrt_ρ1))^2
end

# ╔═╡ 6739ac26-7590-46a8-a75a-2160a1d7c11d
fidelity(ρ, state.𝛒)

# ╔═╡ 00c09086-1eef-4b3f-b764-fe7201e1a354
scatter(d[1, :], d[2, :])

# ╔═╡ be08410c-4fce-405c-9b27-ababf152402e
begin
	d̂ = rand(statê, 4096, IsGaussian)
	scatter(d̂[1, :], d̂[2, :])
end

# ╔═╡ 32c40e60-0989-44f6-b518-9f6203aa6209
function calc_f(; r=(0, 1.8), θ=(0, 2π), n̄=(0, 0.3), dim=700)
	args = SqState.rand_arg_sqth(r, θ, n̄)
	state = SqState.construct_state_sqth(args..., dim)
	argŝ = relu.(m(reshape(Float32.(rand(state, 4096, IsGaussian)), 2, :, 1)))
	statê = SqState.construct_state_sqth(argŝ..., dim)

	fidelity(statê.𝛒, state.𝛒)
end

# ╔═╡ 53da20d3-2d64-4cab-ae4d-0615f750df67
# 5db -> 0.575
# 10db -> 1.151
# 15db -> 1.727

# ╔═╡ bc685a2b-7807-4dd9-b6cd-4a9a1448c09b
sum(calc_f(r=(0., 0.575), dim=100) for _ in 1:100) / 100

# ╔═╡ a4f54b38-ecaa-42b2-bd57-16e9dc52aea0
sum(calc_f(r=(1.15, 1.151), dim=500) for _ in 1:50) / 50

# ╔═╡ 8083f0ef-ad7f-49fe-a248-5931c56ce998
begin
	mse_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	mse_y = [0.997, 0.9922, 0.9839, 0.9770, 0.962,0.9576, 0.9537, 0.9451,0.939, 0.9382]
	cnn_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	cnn_y = [0.999, 0.999, 0.997, 0.997, 0.997, 0.997, 0.996, 0.995, 0.996, 0.992]
	fno_x = [2, 5, 10, 11]
	fno_y = [0.9989662782956508, 0.999431614249688, 0.9996389623744389, 0.9973824329127635]
end

# ╔═╡ 48e7622f-92dd-4a83-8d52-2d0905e87d18
begin
	plotly()
	plot(title="Fidelity of different model", ylabel="Fidelity", xlabel="Squeezing lavel (dB)")
	plot!(mse_x, mse_y, lw=2, label="MLE")
	plot!(cnn_x, cnn_y, lw=2, label="CNN")
	plot!(fno_x, fno_y, lw=2, label="FNO")
	plot!(legend=:bottomright,)
end

# ╔═╡ c7dd44e4-2f9d-43ba-ac59-d959ccf44c79
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# ╔═╡ da114fef-f840-47cb-8b7a-710573151b2e
register(DataDep(
		"QSTDemo",
		"""Quantum state tomography demo dataset""",
		"https://drive.google.com/file/d/1Z6g2ZEhMUhqSEQFebVrilQ2gACRJkTVU/view?usp=sharing",
		fetch_method=gdownload,
		post_fetch_method=unpack
	))

# ╔═╡ 720a99ab-ba50-4fb5-a4c1-499546b73591
function get_data(data_name::String, field::String="data_sq")
    data_file = matopen(data_name)
    data = read(data_file, field)
    close(data_file)

    return data
end

# ╔═╡ 970797a8-f845-48a7-b865-e16255137d95
function sample(data::Matrix, n::Integer)
    data_indices = sort!(rand(1:size(data, 1), n))

	θs = data[data_indices, 2]
	θs .-= minimum(θs)
	θs ./= maximum(θs)
	θs .*= 2π

    return vcat(# 1: x; 2: θ
		reshape(θs, 1, :, 1),
		reshape(data[data_indices, 1], 1, :, 1)
	)
end

# ╔═╡ bc1b37e7-893e-46d2-a3d6-b0e13db987ec
function infer(
    data_name::String;
    n_sample=10, fix_θ=true,
    wf=WignerFunction(LinRange(-3, 3, 101), LinRange(-3, 3, 101), dim=100),
    m=SqState.get_model("model_q2args_sqth_x")
)
    data = get_data(data_name)

    argv = zeros(Float32, 3)
    for _ in 1:n_sample
        argv += relu.(m(sample(data, 4096)))
    end
    argv ./= n_sample

    r, θ, n̄ = argv
    θ = fix_θ ? zero(typeof(θ)) : θ

    state = SqState.construct_state_sqth(r, θ, n̄, wf.m_dim)

    return argv, 𝛒(state), wf(state).𝐰_surface
end

# ╔═╡ a55b5296-1596-4441-ba39-1eff14d5ce54
readdir(datadep"QSTDemo")

# ╔═╡ 983f2809-65f4-40d0-a596-ad8ef54348f9
p = datadep"QSTDemo"

# ╔═╡ f76e6855-b72b-4fbc-8ba4-1f53c16d5746
_, _, w = infer(joinpath(p, "SQ10_5mW.mat"), m=m)

# ╔═╡ f9db36e9-29bc-4d96-9549-beb3049bf356
lim = maximum(abs.(w))

# ╔═╡ 1e8002ec-3371-4cd5-a9b1-93aeb1c29393
surface(w, color=:coolwarm, clim=(-lim, lim), border=:none, title="5mW", camera=(40, 10))

# ╔═╡ Cell order:
# ╠═4655bdc2-1ed7-11ec-022d-7df0b8fdb907
# ╠═661cc8fe-44f3-422a-91f5-96956b748847
# ╠═c55d7af1-0cd3-4b0a-bbd5-0048f15ea851
# ╠═8f82e42f-8c2c-4bd0-ab20-95b0930670dc
# ╠═9588e66c-9c19-4257-8fd2-12455e0c0f0d
# ╠═f0cfd6f3-1e55-46cf-bd58-937e52cb7daa
# ╠═57e75729-4bf4-4e0c-9ef2-bdaf6fb4e22b
# ╠═6739ac26-7590-46a8-a75a-2160a1d7c11d
# ╠═00c09086-1eef-4b3f-b764-fe7201e1a354
# ╠═be08410c-4fce-405c-9b27-ababf152402e
# ╠═32c40e60-0989-44f6-b518-9f6203aa6209
# ╠═53da20d3-2d64-4cab-ae4d-0615f750df67
# ╠═bc685a2b-7807-4dd9-b6cd-4a9a1448c09b
# ╠═a4f54b38-ecaa-42b2-bd57-16e9dc52aea0
# ╠═8083f0ef-ad7f-49fe-a248-5931c56ce998
# ╠═48e7622f-92dd-4a83-8d52-2d0905e87d18
# ╠═510b566f-4fd9-4d96-bfc5-b54385b713ce
# ╠═c7dd44e4-2f9d-43ba-ac59-d959ccf44c79
# ╠═da114fef-f840-47cb-8b7a-710573151b2e
# ╠═720a99ab-ba50-4fb5-a4c1-499546b73591
# ╠═970797a8-f845-48a7-b865-e16255137d95
# ╠═bc1b37e7-893e-46d2-a3d6-b0e13db987ec
# ╠═a55b5296-1596-4441-ba39-1eff14d5ce54
# ╠═983f2809-65f4-40d0-a596-ad8ef54348f9
# ╠═f76e6855-b72b-4fbc-8ba4-1f53c16d5746
# ╠═f9db36e9-29bc-4d96-9549-beb3049bf356
# ╠═1e8002ec-3371-4cd5-a9b1-93aeb1c29393
