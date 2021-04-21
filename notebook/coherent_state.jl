### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 06c04f46-9242-11eb-0cd7-2f11b1b3163c
begin
    using SqState
    using DisplayAs
end

# ╔═╡ e70c7f6c-9241-11eb-1af5-cffa65c64ca5
md"
# Plot Coherent State

JingYu Ning
"

# ╔═╡ a642c114-9242-11eb-0dac-ab5aaeb8b08d
md"
## Initial Wigner function
"

# ╔═╡ 1701e482-9242-11eb-32f2-c1caa541d5f2
begin
    x_range = -10:0.1:10
    p_range = -10:0.1:10
	truncated_photon_number = 100
    wf = WignerFunction(x_range, p_range, dim=truncated_photon_number)
end;

# ╔═╡ b0623b86-9242-11eb-3f51-5bc55266146d
md"
## Rander Wigner function
"

# ╔═╡ 2a6d6c30-9242-11eb-1c85-39b4356bc057
begin
	r = 5
	θ = π/4

	state = CoherentState(Arg(r, θ))
	𝛒 = ρ(state, dim=truncated_photon_number)
	w = wf(𝛒)
end;

# ╔═╡ be689f2c-9242-11eb-0131-8fb39a4495e0
md"
## Plot
"

# ╔═╡ 2ae4b81b-fb69-42d1-a5d4-f92a5f1dd5b4
plot_ρ(𝛒) |> DisplayAs.PNG

# ╔═╡ c9f274d0-9242-11eb-006f-c11362dc2916
md"
**Surface**
"

# ╔═╡ 7abcf782-9242-11eb-353c-4146852588e0
plot_wigner(wf, w, Surface) |> DisplayAs.PNG

# ╔═╡ d23856f0-9242-11eb-06b4-5f6bb54d439c
md"
**Heatmap**
"

# ╔═╡ 8b680bee-9242-11eb-3eb5-05d77644f897
plot_wigner(wf, w, Heatmap) |> DisplayAs.PNG

# ╔═╡ de3267ea-9242-11eb-0ce2-f73d49312529
md"
**Contour**
"

# ╔═╡ 93f085c0-9242-11eb-2589-61d94121bc04
plot_wigner(wf, w, Contour) |> DisplayAs.PNG

# ╔═╡ Cell order:
# ╟─e70c7f6c-9241-11eb-1af5-cffa65c64ca5
# ╠═06c04f46-9242-11eb-0cd7-2f11b1b3163c
# ╟─a642c114-9242-11eb-0dac-ab5aaeb8b08d
# ╠═1701e482-9242-11eb-32f2-c1caa541d5f2
# ╟─b0623b86-9242-11eb-3f51-5bc55266146d
# ╠═2a6d6c30-9242-11eb-1c85-39b4356bc057
# ╟─be689f2c-9242-11eb-0131-8fb39a4495e0
# ╠═2ae4b81b-fb69-42d1-a5d4-f92a5f1dd5b4
# ╟─c9f274d0-9242-11eb-006f-c11362dc2916
# ╠═7abcf782-9242-11eb-353c-4146852588e0
# ╟─d23856f0-9242-11eb-06b4-5f6bb54d439c
# ╠═8b680bee-9242-11eb-3eb5-05d77644f897
# ╟─de3267ea-9242-11eb-0ce2-f73d49312529
# ╠═93f085c0-9242-11eb-2589-61d94121bc04
