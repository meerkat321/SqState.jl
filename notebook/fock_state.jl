### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 5781b9ce-914d-11eb-0dd6-c19da262033c
begin
	using SqState
    using DisplayAs
end

# ╔═╡ 2700b804-914d-11eb-3d64-49496a4560c5
md"
# Plot Wigner Function of Fock State

JingYu Ning
"

# ╔═╡ 46a2899e-914d-11eb-3390-d92ab450c200
md"
## Initial Wigner function
"

# ╔═╡ 4ef3ca9a-914d-11eb-1e78-83bfe054f318
begin
	x_range = -10:0.02:10
	p_range = -10:0.02:10
	wf = WignerFunction(x_range, p_range)
end;

# ╔═╡ 624e0452-914d-11eb-0ad1-cd95271e1db1
md"
## Generate Wigner function and plot
"

# ╔═╡ 607ddac6-914d-11eb-2bdb-bb3af2bc5497
function plot_fock_state(n::Integer, wf::WignerFunction)
    w = wf(ρ(FockState(n)))
	plot_wigner(wf, w, Surface) |> DisplayAs.PNG
end;

# ╔═╡ ba6ac636-914d-11eb-0071-ab847f743e39
plot_fock_state(0, wf)

# ╔═╡ cbc52896-914d-11eb-1b30-3b7c8969b315
plot_fock_state(1, wf)

# ╔═╡ d6f4f0b6-914d-11eb-04be-d5393f28d908
plot_fock_state(2, wf)

# ╔═╡ dcfef6b6-914d-11eb-3b9e-13da0d651cba
plot_fock_state(3, wf)

# ╔═╡ 7217e9f6-914e-11eb-2203-a99fa64f085f
plot_fock_state(4, wf)

# ╔═╡ 77f5b092-914e-11eb-066b-5f8f6f38aeec
plot_fock_state(5, wf)

# ╔═╡ Cell order:
# ╟─2700b804-914d-11eb-3d64-49496a4560c5
# ╠═5781b9ce-914d-11eb-0dd6-c19da262033c
# ╟─46a2899e-914d-11eb-3390-d92ab450c200
# ╠═4ef3ca9a-914d-11eb-1e78-83bfe054f318
# ╟─624e0452-914d-11eb-0ad1-cd95271e1db1
# ╠═607ddac6-914d-11eb-2bdb-bb3af2bc5497
# ╠═ba6ac636-914d-11eb-0071-ab847f743e39
# ╠═cbc52896-914d-11eb-1b30-3b7c8969b315
# ╠═d6f4f0b6-914d-11eb-04be-d5393f28d908
# ╠═dcfef6b6-914d-11eb-3b9e-13da0d651cba
# ╠═7217e9f6-914e-11eb-2203-a99fa64f085f
# ╠═77f5b092-914e-11eb-066b-5f8f6f38aeec
