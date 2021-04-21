### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ d393a05a-6532-11eb-04b2-35f9af4bdbc2
begin
	using SqState
	using BenchmarkTools
	using DataDeps
	using DisplayAs
end

# ╔═╡ 2279cfae-6534-11eb-04bd-e7ce1910fc83
md"
# Plot Wigner Function

JingYu Ning
"

# ╔═╡ e46bc790-6532-11eb-2c16-3b43640c7653
md"
## Initial Wigner function
"

# ╔═╡ f7af8256-6532-11eb-066b-d55e91cddc92
begin
	x_range = -10:0.1:10
	p_range = -10:0.1:10
	wf = WignerFunction(x_range, p_range)
end;

# ╔═╡ 2fe499fe-6533-11eb-0bcd-9d00d9b4e1d0
md"
## Render Wigner function

The squeezed state data showing bellow is measured in Prof. RK Lee's Lab.
"

# ╔═╡ 125c6e66-6533-11eb-03b4-7122cc3e5806
begin
	# read from HDF5 file:
	data_path = datadep"SqState/data/dm.h5"
    data_name = "SQ4"
    𝛒 = read_ρ(data_path, data_name)
    w = wf(𝛒)
end;

# ╔═╡ ce5a611a-689c-11eb-0bc0-11a765ac2ffa
@benchmark wf($𝛒)

# ╔═╡ 6cb3a712-6533-11eb-34f3-6339e020be33
md"
## Plot
"

# ╔═╡ dedb97c8-33f5-43b7-ba04-367d3fe32351
md"
**Density Matrix**
"

# ╔═╡ eae5423e-417e-40e5-80b9-62956b351b17
plot_ρ(𝛒) |> DisplayAs.PNG

# ╔═╡ 668e0075-fead-4d0f-bc49-5dccc64989bc
@benchmark plot_ρ($𝛒)

# ╔═╡ 973b2490-688c-11eb-0c2a-39e621770fba
md"
**Surface**
"

# ╔═╡ a26f33b0-688c-11eb-246c-6f2a4c09f09a
plot_wigner(wf, w, Surface) |> DisplayAs.PNG

# ╔═╡ a9db6a98-689a-11eb-11d5-d3fea7c24256
@benchmark plot_wigner($wf, $w, Surface)

# ╔═╡ a24789e8-6533-11eb-2bb9-db79fb1c365c
md"
**Heatmap**
"

# ╔═╡ 7f9264d6-6533-11eb-1c6c-434909802cb5
plot_wigner(wf, w, Heatmap) |> DisplayAs.PNG

# ╔═╡ b581cdf6-689a-11eb-209c-45745b570e50
@benchmark plot_wigner($wf, $w, Heatmap)

# ╔═╡ 903502b0-6533-11eb-2ed0-0d1bcfe1fe99
md"
**Contour**
"

# ╔═╡ 0da7f0fc-6538-11eb-0aa7-e1635323a04d
plot_wigner(wf, w, Contour) |> DisplayAs.PNG

# ╔═╡ b82b6e38-689a-11eb-0741-a332b072ea1f
@benchmark plot_wigner($wf, $w, Contour)

# ╔═╡ Cell order:
# ╟─2279cfae-6534-11eb-04bd-e7ce1910fc83
# ╠═d393a05a-6532-11eb-04b2-35f9af4bdbc2
# ╟─e46bc790-6532-11eb-2c16-3b43640c7653
# ╠═f7af8256-6532-11eb-066b-d55e91cddc92
# ╟─2fe499fe-6533-11eb-0bcd-9d00d9b4e1d0
# ╠═125c6e66-6533-11eb-03b4-7122cc3e5806
# ╠═ce5a611a-689c-11eb-0bc0-11a765ac2ffa
# ╟─6cb3a712-6533-11eb-34f3-6339e020be33
# ╟─dedb97c8-33f5-43b7-ba04-367d3fe32351
# ╠═eae5423e-417e-40e5-80b9-62956b351b17
# ╠═668e0075-fead-4d0f-bc49-5dccc64989bc
# ╟─973b2490-688c-11eb-0c2a-39e621770fba
# ╠═a26f33b0-688c-11eb-246c-6f2a4c09f09a
# ╠═a9db6a98-689a-11eb-11d5-d3fea7c24256
# ╟─a24789e8-6533-11eb-2bb9-db79fb1c365c
# ╠═7f9264d6-6533-11eb-1c6c-434909802cb5
# ╠═b581cdf6-689a-11eb-209c-45745b570e50
# ╟─903502b0-6533-11eb-2ed0-0d1bcfe1fe99
# ╠═0da7f0fc-6538-11eb-0aa7-e1635323a04d
# ╠═b82b6e38-689a-11eb-0741-a332b072ea1f
