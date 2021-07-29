### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 4daac2cf-dd9c-4bff-861a-fdcb828df03c
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ b673d52b-0d39-4dba-b2f6-6d3c2c207e3a
begin
	using BenchmarkTools
	using QuantumStateBase
	using QuantumStatePlots
	using Plots
	wf = WignerFunction(-10:0.1:10, -10:0.1:10)
end;

# ╔═╡ 79ca1104-0218-4426-8722-0024b92a0eef
md"
# Quantum State Base

JingYu Ning
"

# ╔═╡ 4d8b01d4-dcbe-4d1a-9c22-da4e581d8143
md"
## Fock state

Fock state is thhe common basis for quantum mechanics. In this prokect, we construct most of the states in Fock basis. For example, `CoherentState`, `SqueezedState` and `SqueezedThermalState` etc.
"

# ╔═╡ 16771043-30ec-4d89-8ff7-46161f33b8b3
FockState(0)

# ╔═╡ c31bba4f-e23e-4f26-aae7-c02c79f4f6b5
plot_wigner(wf(FockState(0)), Contour)

# ╔═╡ 4974bb09-0901-4721-ac9e-37269c0242b1
md"
Also, the `QuantumStateBase` accept various alias like `VacuumState`, `SinglePhotonState`, and `NumberState` etc.
"

# ╔═╡ b0c1c790-8dcc-4d76-803c-bf6a3bd61d95
vec(NumberState(5)) == vec(FockState(5))

# ╔═╡ 4a26a842-9907-49df-a954-ced09352f39d
vec(VacuumState()) == vec(FockState(0))

# ╔═╡ ac4e9d5c-ae31-4b60-a1d7-21017db9da2d
vec(SinglePhotonState()) == vec(FockState(1))

# ╔═╡ 4dce2059-882d-40b1-b2ec-31c9e44307aa
md"
## Coherent state

Coherent state is defined as:

$| \alpha \rangle = exp(-\frac{1}{2}|\alpha|^2)
\sum^\infty_{n=0} \frac{\alpha^n}{\sqrt{n!}} | n \rangle$

We can therefore define displacement operator and implies that

$| \alpha \rangle = \hat{D}(\alpha) | 0 \rangle$

One can use the pre-defined constructor to declare a coherent state.
"

# ╔═╡ ab919e88-14b8-4ff8-896c-dcfe706a0ef7
CoherentState(α(5., π/4))

# ╔═╡ d6ad648e-be5e-408e-a93b-3b24e82001d2
md"
It is also recommend to construct a coherent state by apply the displacement operator onto a vacuum state
"

# ╔═╡ e61381a9-1291-4bd9-adb9-584128607060
displace!(VacuumState(), α(5., π/4))

# ╔═╡ f20ea730-270d-4a61-a8ba-11fbc6e38358
plot_wigner(wf(CoherentState(α(5., π/4))), Contour)

# ╔═╡ 3fbb03ea-5a92-4589-9748-63e6a0486240
md"
## Squeezed state

Squeezed state is defined as:

$exp(\frac{1}{2} (\xi^* \hat{a}^2 - \xi \hat{a}^{\dagger 2})) | 0 \rangle$

We implies Squeezing operator

$\hat{S}(\xi) = exp(\frac{1}{2} (\xi^* \hat{a}^2 - \xi \hat{a}^{\dagger 2}))$

One can use the pre-defined constructor to declare a squeezed state.
"

# ╔═╡ d5cf7b3f-c9a2-4450-8bee-cf510100968f
SqueezedState(ξ(0.8, π/4))

# ╔═╡ 360282a0-47e1-4501-ae88-ce08c5958c7a
md"
It is also recommend to construct a squeezed state by apply the squeezing operator onto a vacuum state
"

# ╔═╡ ed4b3452-e969-4e37-a9d0-aaec89e59120
squeeze!(VacuumState(), ξ(0.8, π/4))

# ╔═╡ 1c765e8a-c47e-4133-87ac-810a55c257c9
plot_wigner(wf(SqueezedState(ξ(0.8, π/4))), Contour)

# ╔═╡ 5e5052e9-9745-40f2-99fa-7eed9e3d5197
md"
## Other State

There are numerous of pre-defined constructor to construct state. For example, `ThermalState`, `SqueezedThermalState` etc.

The most powerful featuure is that one can construct their own state by operators such as $a$ and $a^{\dagger}$, $\hat{S}$ and $\hat{D}$ and so on.

The following sectioon is about constructing a cat state.
"

# ╔═╡ 85eee830-aa57-45b1-b412-83faa69c0508
md"
## Cat State in single mode

Cat state is the superposition of two coherent state with opposite phase.

The following formulas are coherent states defined in the Fock state.

$\begin{align*}
| \alpha \rangle &= exp(-\frac{1}{2}|\alpha|^2)
\sum^\infty_{n=0} \frac{\alpha^n}{\sqrt{n!}} | n \rangle \\

&= \hat{D}(\alpha) | 0 \rangle \\

| -\alpha \rangle &= exp(-\frac{1}{2}|-\alpha|^2)
\sum^\infty_{n=0} \frac{(-\alpha)^n}{\sqrt{n!}} | n \rangle \\

&= \hat{D}(-\alpha) | 0 \rangle
\end{align*}$

The even cat state is therefore defined as:

$| cat_e \rangle = | \alpha \rangle + | -\alpha \rangle$

And odd cat state is therefore defined as:

$| cat_o \rangle = | \alpha \rangle - | -\alpha \rangle$
"

# ╔═╡ 92f0daf2-ee46-11eb-27f0-5f90c0bcb61d
begin
	αₚ = displace!(VacuumState(), α(2., π/4))
	αₙ = displace!(VacuumState(), α(-2., π/4))

	catₑ = StateVector(vec(αₚ) + vec(αₙ), QuantumStateBase.DIM)
	catₒ = StateVector(vec(αₚ) - vec(αₙ), QuantumStateBase.DIM)
end;

# ╔═╡ 7c5728ce-5d9e-4f9f-b036-baa09b34b310
plot_wigner(wf(catₑ), Contour)

# ╔═╡ 8d56e643-aef3-426e-b098-bdb45a6f20b3
plot_wigner(wf(catₒ), Contour)

# ╔═╡ 06cc70ba-3de8-40b2-9ace-00ab2c8656dd
begin
	cat_points = rand(catₑ, 4096)

	scatter(
		cat_points[1, :], cat_points[2, :],
		title="Even cat state", xlabel="θ", ylabel="x", legend=false,
		ylim=(-5, 5), size=(800, 400),
	)
end

# ╔═╡ b39f1784-95ec-4529-b922-fe6bed27eae2
md"
## The blazing fast non-Gaussian data sampler

It is always a difficult topic when it comes to sampling data from arbitrary prabability density function.

In this project, we introduse a non-Gaussian prabability density function sampler. The sampler is implemented by rejection method. And can sample about 4000 points from a pdf in few seconds.
"

# ╔═╡ 1b8d62a2-3bdc-4a22-9f21-bbb5198f8b7b
@benchmark rand(catₑ, 4096)

# ╔═╡ Cell order:
# ╟─79ca1104-0218-4426-8722-0024b92a0eef
# ╟─4daac2cf-dd9c-4bff-861a-fdcb828df03c
# ╠═b673d52b-0d39-4dba-b2f6-6d3c2c207e3a
# ╟─4d8b01d4-dcbe-4d1a-9c22-da4e581d8143
# ╠═16771043-30ec-4d89-8ff7-46161f33b8b3
# ╠═c31bba4f-e23e-4f26-aae7-c02c79f4f6b5
# ╟─4974bb09-0901-4721-ac9e-37269c0242b1
# ╠═b0c1c790-8dcc-4d76-803c-bf6a3bd61d95
# ╠═4a26a842-9907-49df-a954-ced09352f39d
# ╠═ac4e9d5c-ae31-4b60-a1d7-21017db9da2d
# ╟─4dce2059-882d-40b1-b2ec-31c9e44307aa
# ╠═ab919e88-14b8-4ff8-896c-dcfe706a0ef7
# ╟─d6ad648e-be5e-408e-a93b-3b24e82001d2
# ╠═e61381a9-1291-4bd9-adb9-584128607060
# ╠═f20ea730-270d-4a61-a8ba-11fbc6e38358
# ╟─3fbb03ea-5a92-4589-9748-63e6a0486240
# ╠═d5cf7b3f-c9a2-4450-8bee-cf510100968f
# ╟─360282a0-47e1-4501-ae88-ce08c5958c7a
# ╠═ed4b3452-e969-4e37-a9d0-aaec89e59120
# ╠═1c765e8a-c47e-4133-87ac-810a55c257c9
# ╟─5e5052e9-9745-40f2-99fa-7eed9e3d5197
# ╟─85eee830-aa57-45b1-b412-83faa69c0508
# ╠═92f0daf2-ee46-11eb-27f0-5f90c0bcb61d
# ╠═7c5728ce-5d9e-4f9f-b036-baa09b34b310
# ╠═8d56e643-aef3-426e-b098-bdb45a6f20b3
# ╠═06cc70ba-3de8-40b2-9ace-00ab2c8656dd
# ╟─b39f1784-95ec-4529-b922-fe6bed27eae2
# ╠═1b8d62a2-3bdc-4a22-9f21-bbb5198f8b7b
