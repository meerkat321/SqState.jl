### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ ea40dcf0-b5f0-11eb-32a7-3123ff0c686e
using LinearAlgebra

# ╔═╡ 3b81a4ac-aeb8-4ca2-8dc0-e8ac5275faa5
using SqState

# ╔═╡ a532dc4c-5a28-471c-9ecf-07dcc7b74634
creation = diagm(-1 => sqrt.(1:34))

# ╔═╡ b61d59e8-d501-4c9c-9da4-1587a02cd001
annihilation = diagm(1 => sqrt.(1:34))

# ╔═╡ 86913628-6c58-4c6a-999b-74344fef488e
begin
	vacuum = zeros(35)
	vacuum[1] = 1
	vacuum
end

# ╔═╡ 66d6c439-0a15-41aa-80d9-702e4a462738
begin
	r = 5
	θ = π/4

	D = exp(r * exp(im * θ) * creation - r * exp(-im * θ) * annihilation)
end

# ╔═╡ 97dedde4-b051-4765-957b-42e8dfa77c73
a = D * vacuum

# ╔═╡ e31919b6-5df4-4bf3-9256-2f02ede9a51f
ρa = a * a'

# ╔═╡ e2f1cbaa-3c37-4fb1-92fe-42aeeced0cc4
wf = WignerFunction(-10:0.1:10, -10:0.1:10);

# ╔═╡ 0005f3a6-e871-45da-b5b9-9773802485f6
w = wf(ρa);

# ╔═╡ e7950ab3-4e4c-445e-a17f-dc0b3630d24f
plot_wigner(wf, w, Surface)

# ╔═╡ Cell order:
# ╠═ea40dcf0-b5f0-11eb-32a7-3123ff0c686e
# ╠═a532dc4c-5a28-471c-9ecf-07dcc7b74634
# ╠═b61d59e8-d501-4c9c-9da4-1587a02cd001
# ╠═86913628-6c58-4c6a-999b-74344fef488e
# ╠═66d6c439-0a15-41aa-80d9-702e4a462738
# ╠═97dedde4-b051-4765-957b-42e8dfa77c73
# ╠═e31919b6-5df4-4bf3-9256-2f02ede9a51f
# ╠═3b81a4ac-aeb8-4ca2-8dc0-e8ac5275faa5
# ╠═e2f1cbaa-3c37-4fb1-92fe-42aeeced0cc4
# ╠═0005f3a6-e871-45da-b5b9-9773802485f6
# ╠═e7950ab3-4e4c-445e-a17f-dc0b3630d24f
