### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ d393a05a-6532-11eb-04b2-35f9af4bdbc2
begin
	using SqState
	using DataDeps
	using HDF5
	using LinearAlgebra
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
	x_range = -10:0.02:10
	p_range = -10:0.02:10
	wf = WignerFunction(x_range, p_range)
end;

# ╔═╡ 2fe499fe-6533-11eb-0bcd-9d00d9b4e1d0
md"
## Post-processing for data inferenced by model

The squeezed state data showing bellow is measured in Prof. RK Lee's Lab.
"

# ╔═╡ 93bca615-e9e8-4cd6-bc16-19cd7fe9aea8
function read_model_inference(data_path)
	real_part = h5open(data_path, "r") do file
        read(file, "real_part")
    end
    imag_part = h5open(data_path, "r") do file
        read(file, "imaginary_part")
    end

    return complex.(real_part, imag_part)[:, 1]
end

# ╔═╡ ffcb6baf-2be3-4f71-9ab0-65e09261aaa9
function reshape_model_inference(data::Vector{ComplexF64})
	l_ch = zeros(35, 35)
	start_i = 1
	for i in -34:0
		l_ch += diagm(i => data[start_i: start_i+34+i])
		start_i += 35+i
	end

	return l_ch
end

# ╔═╡ 6cb3a712-6533-11eb-34f3-6339e020be33
md"
## Plot
"

# ╔═╡ 125c6e66-6533-11eb-03b4-7122cc3e5806
function render_data(file_name)
	# read from HDF5 file:
	data_path = datadep"SqState/data/data0406"

	# post-processing
    l_ch = reshape_model_inference(
		read_model_inference(joinpath(data_path, file_name))
	)
	𝛒 = l_ch * l_ch'

	# render Wigner
    w = wf(𝛒)

	# plot
	plot_all(wf, w, 𝛒, levels=10)
end

# ╔═╡ 094eeba7-d4ab-4c0e-918a-b84c036cf874
md"
#### SQ1
"

# ╔═╡ 72234876-51a5-48be-93f1-56e7ffd8e614
render_data("SQ1_model_output.h5") |> DisplayAs.PNG

# ╔═╡ d4394d3a-6d7d-4323-9318-51f0ff7737e6
md"
#### SQ2
"

# ╔═╡ dc6cb4db-7e8d-47cd-922a-0a8f6ceb5e83
render_data("SQ2_model_output.h5") |> DisplayAs.PNG

# ╔═╡ aa231ec1-403a-40ea-88d4-9f3cd0d463de
md"
#### SQ3
"

# ╔═╡ 598eca83-ea07-4fb7-9765-f523eb01d869
render_data("SQ3_model_output.h5") |> DisplayAs.PNG

# ╔═╡ 5c9f5dea-ac7f-43cc-a375-2dd0dcd74115
md"
#### SQ4
"

# ╔═╡ c55609ce-d683-4d94-b3bb-7798afb68178
render_data("SQ4_model_output.h5") |> DisplayAs.PNG

# ╔═╡ Cell order:
# ╟─2279cfae-6534-11eb-04bd-e7ce1910fc83
# ╠═d393a05a-6532-11eb-04b2-35f9af4bdbc2
# ╟─e46bc790-6532-11eb-2c16-3b43640c7653
# ╠═f7af8256-6532-11eb-066b-d55e91cddc92
# ╟─2fe499fe-6533-11eb-0bcd-9d00d9b4e1d0
# ╠═93bca615-e9e8-4cd6-bc16-19cd7fe9aea8
# ╠═ffcb6baf-2be3-4f71-9ab0-65e09261aaa9
# ╟─6cb3a712-6533-11eb-34f3-6339e020be33
# ╠═125c6e66-6533-11eb-03b4-7122cc3e5806
# ╟─094eeba7-d4ab-4c0e-918a-b84c036cf874
# ╠═72234876-51a5-48be-93f1-56e7ffd8e614
# ╟─d4394d3a-6d7d-4323-9318-51f0ff7737e6
# ╠═dc6cb4db-7e8d-47cd-922a-0a8f6ceb5e83
# ╟─aa231ec1-403a-40ea-88d4-9f3cd0d463de
# ╠═598eca83-ea07-4fb7-9765-f523eb01d869
# ╟─5c9f5dea-ac7f-43cc-a375-2dd0dcd74115
# ╠═c55609ce-d683-4d94-b3bb-7798afb68178
