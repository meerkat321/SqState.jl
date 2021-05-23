### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 59cc9197-f5c7-478a-9583-c783d6a9e264
using SqState

# ╔═╡ 311df460-bb95-11eb-3eb1-ab75873431b0
md"
# SqState Tutorial

JingYu Ning
"

# ╔═╡ e175c58a-f5fa-4ace-b1d9-6a1849f3eba4
md"
## Quantum states
"

# ╔═╡ 0dc06b42-68b4-4184-8e89-cfea7b2be1d6
md"
### Pure state
"

# ╔═╡ e99fe10b-daf6-4e4c-b7db-c2c1e2bcea60
md"
#### Fock state
"

# ╔═╡ 1ad47d0b-ecb2-4f8e-8873-c03fc6fa1b0e
vec(FockState(3))

# ╔═╡ 84b004c7-722f-4191-80bd-ae330ff9ae04
𝛒(FockState(3))

# ╔═╡ 5d321eda-af61-415f-adbb-98b740fb3afa
md"
#### Number state
"

# ╔═╡ 8d51ef92-2f87-456e-96a1-1115a6c1294c
vec(NumberState(4))

# ╔═╡ 3d1dff7f-9036-435d-86c9-10f5864c3c26
𝛒(NumberState(4))

# ╔═╡ 261764ac-a142-41e2-89e4-712a6e653421
md"
#### Vacuum state
"

# ╔═╡ a2f52f38-e27f-4b04-8f9f-fe798a74a7f4
vec(VacuumState())

# ╔═╡ cdceb853-034d-4524-82e0-7b0ce1c5be78
𝛒(VacuumState())

# ╔═╡ 91a76002-f54e-4194-bdf3-a7d8ddb89ae7
md"
#### Single photon state
"

# ╔═╡ 539161d4-bf0b-4e62-bbe0-8cc11154d309
vec(SinglePhotonState())

# ╔═╡ 17e524df-7c62-4040-80d9-22a1c978db9e
𝛒(SinglePhotonState())

# ╔═╡ c97a8228-11cf-4ac2-b980-15e7e4f2f0e4
md"
#### Coherent state
"

# ╔═╡ 8e9edb98-d614-4007-991c-e5cddce0f2cf
vec(CoherentState(α(2., π/4)))

# ╔═╡ d6a32787-2c23-41b6-be91-26b1b917e23f
𝛒(CoherentState(α(2., π/4)))

# ╔═╡ 1fc47064-fec8-4d86-8bce-3878b4f9d970
md"
#### Squeezed state
"

# ╔═╡ c999ce9f-9d43-46f0-9cf7-2351c8b2db31
vec(SqueezedState(ξ(1., π/8)))

# ╔═╡ 8348466f-2caa-44f1-b094-5508645feb73
𝛒(SqueezedState(ξ(1., π/8)))

# ╔═╡ a5073f9a-5896-4b5b-8444-2cbc6dab7434
md"
### Mixed state
"

# ╔═╡ 956b19e2-b54c-4d02-be74-3910a322abc5
md"
#### Thermal state
"

# ╔═╡ cc842044-0914-491b-828b-8616b5b86088
𝛒(ThermalState(0.3))

# ╔═╡ 80db7f67-ac20-4e73-858d-d06b1f09e96e
md"
#### Squeezed thermal state
"

# ╔═╡ 7f1ca0d5-f34a-4d9f-bca7-f7ce8b8a3a2a
𝛒(SqueezedThermalState(ξ(1., π/8), 0.3))

# ╔═╡ e3d77152-fb07-4868-b3a0-69c2be0fca88
md"
## Operators
"

# ╔═╡ fa54e0e0-2177-42cb-8d43-a29d230743fb
md"
### a† and a
"

# ╔═╡ 569bc14e-7152-4c6e-8682-c3caa6df981c
vec(create!(VacuumState()))

# ╔═╡ 118f84b9-942f-4d7e-a893-10a84f0cd4a3
vec(annihilate!(SinglePhotonState()))

# ╔═╡ 1db0bf2d-07ae-4626-acb8-f735b60a949b
md"
### Displacement
"

# ╔═╡ 9c35f27f-4bac-4073-991e-22f8b88a307a
vec(displace!(VacuumState(), α(2., π/4)))

# ╔═╡ b993d819-bdc5-41e1-959d-502c2186cd0a
md"
### Squeezing
"

# ╔═╡ c2f7eb89-37ab-4131-bea2-50ddb68e95e3
vec(squeeze!(VacuumState(), ξ(1., π/8)))

# ╔═╡ a4cf8c2d-e363-4730-aa9b-5f55af35cb25
𝛒(squeeze!(ThermalState(0.3), ξ(1., π/8)))

# ╔═╡ 2fdb2e22-bfe1-4be9-a7c2-f45c4537f575
md"
## Plot
"

# ╔═╡ 80d3bcc6-640d-4f30-8fc4-030eaf5b6908
md"
**Initial Wigner function**
"

# ╔═╡ b9df172e-31a8-4343-b970-0da7fc29721c
wf = WignerFunction(-10:0.1:10, -10:0.1:10);

# ╔═╡ d27bc2c9-7f5e-4ada-bcc6-53747af89207
md"
### Plot Wigner function
"

# ╔═╡ 2a1d78be-2af2-41f3-ae25-4546106b3fe5
plot_wigner(wf(VacuumState()), Surface)

# ╔═╡ c4aa5e5f-7fd3-43d9-a614-431af489c7c7
plot_wigner(wf(SinglePhotonState()), Surface)

# ╔═╡ f2591ec9-e920-46da-b006-198a3a8048b3
plot_wigner(wf(CoherentState(α(2., 3π/4))), Surface)

# ╔═╡ c3ab3465-8a57-4559-8435-2b7de636c011
plot_wigner(wf(SqueezedState(ξ(0.8, 1π/8))), Surface)

# ╔═╡ 46f49def-5730-48f0-8ccc-b7e6f2d6c768
plot_wigner(wf(ThermalState(0.3)), Surface)

# ╔═╡ 119b2861-6022-4390-82a0-1b3801b41e0d
plot_wigner(wf(SqueezedThermalState(ξ(0.8, 1π/8), 0.3)), Surface)

# ╔═╡ 4d85b132-2497-412c-8ad0-634653028352
plot_wigner(wf(SqueezedThermalState(ξ(0.8, 1π/8), 0.3)), Contour)

# ╔═╡ 77a29b92-ca22-4800-addf-7161dd83d91c
plot_wigner(wf(SqueezedThermalState(ξ(0.8, 1π/8), 0.3)), Heatmap)

# ╔═╡ 12ee4941-f37c-4e82-bede-eb437a93fb81
md"
### Plot density matrix
"

# ╔═╡ a6f24457-a72f-4ee9-aadf-925121ca5642
plot_ρ(VacuumState())

# ╔═╡ 5458b817-6c6a-4836-98b2-fa267d6aa07e
plot_ρ(CoherentState(α(2., 3π/4)))

# ╔═╡ db7fdba2-0825-495f-83ba-ca264e58f46e
plot_ρ(SqueezedState(ξ(0.8, 1π/8)))

# ╔═╡ 1aeff5e5-ac8d-4c45-b745-4b7ea5ae4fab
plot_ρ(SqueezedThermalState(ξ(0.8, 1π/8), 0.3))

# ╔═╡ 0afb0b14-fa9c-4ba9-bd4d-31e421331937
plot_ρ(ThermalState(0.3))

# ╔═╡ Cell order:
# ╟─311df460-bb95-11eb-3eb1-ab75873431b0
# ╠═59cc9197-f5c7-478a-9583-c783d6a9e264
# ╟─e175c58a-f5fa-4ace-b1d9-6a1849f3eba4
# ╟─0dc06b42-68b4-4184-8e89-cfea7b2be1d6
# ╟─e99fe10b-daf6-4e4c-b7db-c2c1e2bcea60
# ╠═1ad47d0b-ecb2-4f8e-8873-c03fc6fa1b0e
# ╠═84b004c7-722f-4191-80bd-ae330ff9ae04
# ╟─5d321eda-af61-415f-adbb-98b740fb3afa
# ╠═8d51ef92-2f87-456e-96a1-1115a6c1294c
# ╠═3d1dff7f-9036-435d-86c9-10f5864c3c26
# ╟─261764ac-a142-41e2-89e4-712a6e653421
# ╠═a2f52f38-e27f-4b04-8f9f-fe798a74a7f4
# ╠═cdceb853-034d-4524-82e0-7b0ce1c5be78
# ╟─91a76002-f54e-4194-bdf3-a7d8ddb89ae7
# ╠═539161d4-bf0b-4e62-bbe0-8cc11154d309
# ╠═17e524df-7c62-4040-80d9-22a1c978db9e
# ╟─c97a8228-11cf-4ac2-b980-15e7e4f2f0e4
# ╠═8e9edb98-d614-4007-991c-e5cddce0f2cf
# ╠═d6a32787-2c23-41b6-be91-26b1b917e23f
# ╟─1fc47064-fec8-4d86-8bce-3878b4f9d970
# ╠═c999ce9f-9d43-46f0-9cf7-2351c8b2db31
# ╠═8348466f-2caa-44f1-b094-5508645feb73
# ╟─a5073f9a-5896-4b5b-8444-2cbc6dab7434
# ╟─956b19e2-b54c-4d02-be74-3910a322abc5
# ╠═cc842044-0914-491b-828b-8616b5b86088
# ╟─80db7f67-ac20-4e73-858d-d06b1f09e96e
# ╠═7f1ca0d5-f34a-4d9f-bca7-f7ce8b8a3a2a
# ╟─e3d77152-fb07-4868-b3a0-69c2be0fca88
# ╟─fa54e0e0-2177-42cb-8d43-a29d230743fb
# ╠═569bc14e-7152-4c6e-8682-c3caa6df981c
# ╠═118f84b9-942f-4d7e-a893-10a84f0cd4a3
# ╟─1db0bf2d-07ae-4626-acb8-f735b60a949b
# ╠═9c35f27f-4bac-4073-991e-22f8b88a307a
# ╟─b993d819-bdc5-41e1-959d-502c2186cd0a
# ╠═c2f7eb89-37ab-4131-bea2-50ddb68e95e3
# ╠═a4cf8c2d-e363-4730-aa9b-5f55af35cb25
# ╟─2fdb2e22-bfe1-4be9-a7c2-f45c4537f575
# ╟─80d3bcc6-640d-4f30-8fc4-030eaf5b6908
# ╠═b9df172e-31a8-4343-b970-0da7fc29721c
# ╟─d27bc2c9-7f5e-4ada-bcc6-53747af89207
# ╠═2a1d78be-2af2-41f3-ae25-4546106b3fe5
# ╠═c4aa5e5f-7fd3-43d9-a614-431af489c7c7
# ╠═f2591ec9-e920-46da-b006-198a3a8048b3
# ╠═c3ab3465-8a57-4559-8435-2b7de636c011
# ╠═46f49def-5730-48f0-8ccc-b7e6f2d6c768
# ╠═119b2861-6022-4390-82a0-1b3801b41e0d
# ╠═4d85b132-2497-412c-8ad0-634653028352
# ╠═77a29b92-ca22-4800-addf-7161dd83d91c
# ╟─12ee4941-f37c-4e82-bede-eb437a93fb81
# ╠═a6f24457-a72f-4ee9-aadf-925121ca5642
# ╠═5458b817-6c6a-4836-98b2-fa267d6aa07e
# ╠═db7fdba2-0825-495f-83ba-ca264e58f46e
# ╠═1aeff5e5-ac8d-4c45-b745-4b7ea5ae4fab
# ╠═0afb0b14-fa9c-4ba9-bd4d-31e421331937
