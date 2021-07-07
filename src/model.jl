using SqState
using LinearAlgebra
using Flux
using CUDA
using JLD2

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(true)
end

dim = 70

function conv_layers(ch::NTuple{4, <:Integer}, kernel_size::NTuple{3, <:Integer})
    return [
        Conv((kernel_size[1], ), ch[1]=>ch[2], pad=SamePad()),
        BatchNorm(ch[2], relu),
        Conv((kernel_size[2], ), ch[2]=>ch[3], pad=SamePad()),
        BatchNorm(ch[3], relu),
        Conv((kernel_size[3], ), ch[3]=>ch[4], pad=SamePad()),
        BatchNorm(ch[4], relu),
    ]
end

function residual_block()
    convs = conv_layers((128, 64, 64, 128), (1, 4, 1))

    return [
        x -> Chain(convs...)(x) + x,
        MeanPool((2, ))
    ]
end

function model()
    res_blk = vcat([residual_block() for _ = 1:10]...)

    return Chain(
        Conv((4, ), 1=>128, relu, pad=SamePad()),
        Chain(res_blk...),
        flatten,
        Dense(4*128, 2048),
        Dense(2048, 2*dim*dim)
    )
end

m = model()

function loss(x, 𝐲)
    l̂ = m(x)
    𝐥̂_real = reshape(l̂[1:(dim*dim)], (dim, dim))
    𝐥̂_imag = reshape(l̂[(dim*dim+1):end], (dim, dim))

    𝐥̂ = 𝐥̂_real + im * 𝐥̂_imag
    𝛒̂ = 𝐥̂ * 𝐥̂'
    𝛒̂ = hcat(real.(𝛒̂), imag.(𝛒̂))

    # l ∈ (dim, n)
    # l * l' ∈ (dim, dim) # positive semi-definite matrix
    # Flux.mse(l * l', 𝐲)

    return Flux.mse(𝛒̂, 𝐲)
end

# DEBUG: remove complex relative operation
function loss_y(l̂, 𝐲)
    # l̂ = m(x)
    𝐥̂_real = reshape(l̂[1:(dim*dim)], (dim, dim))
    𝐥̂_imag = reshape(l̂[(dim*dim+1):end], (dim, dim))

    𝐥̂ = 𝐥̂_real + im * 𝐥̂_imag
    𝛒̂ = 𝐥̂ * 𝐥̂'
    𝛒̂ = hcat(real.(𝛒̂), imag.(𝛒̂))

    # l ∈ (dim, n)
    # l * l' ∈ (dim, dim) # positive semi-definite matrix
    # Flux.mse(l * l', 𝐲)

    return Flux.mse(𝛒̂, 𝐲)
end

file_names = readdir(SqState.training_data_path())
f = jldopen(joinpath(SqState.training_data_path(), file_names[1]), "r")
points = f["points"]
𝛒s = f["𝛒s"]

for i in 1:1 # 10000
    x = reshape(Float32.(points[:, i]), (4096, 1, 1)) # 4096 points 1 channel, 1 data in a batch
    𝐲 = ComplexF32.(𝛒s[i])
    𝐲 = hcat(real.(𝐲), imag.(𝐲))

    @show size(reshape(m(x), :))
    @show loss(x, 𝐲)
    # @show gradient(x->sum(m(x)), x)
    # @show gradient(x->loss(x, 𝐲), x)
    @show gradient(l̂->loss_y(l̂, 𝐲), rand(Float32, 2*dim*dim))
end
