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

function loss(l̂, 𝐲)
    𝐥̂_real = reshape(l̂[1:(dim*dim)], (dim, dim))
    𝐥̂_imag = reshape(l̂[(dim*dim+1):end], (dim, dim))

    𝐥̂ = 𝐥̂_real + im * 𝐥̂_imag

    # l ∈ (dim, n)
    # l * l' ∈ (dim, dim) # positive semi-definite matrix
    # Flux.mse(l * l', 𝐲)

    return Flux.mse(𝐥̂ * 𝐥̂', 𝐲)
end

function conv_layers(ch::NTuple{4, <:Integer}, kernel_size::NTuple{3, <:Integer})
    return Chain(
        Conv((kernel_size[1], ), ch[1]=>ch[2], pad=SamePad()),
        BatchNorm(ch[2], relu),
        Conv((kernel_size[2], ), ch[2]=>ch[3], pad=SamePad()),
        BatchNorm(ch[3], relu),
        Conv((kernel_size[3], ), ch[3]=>ch[4], pad=SamePad()),
        BatchNorm(ch[4], relu),
    )
end

function residual_block()
    return Chain(
        x -> conv_layers((128, 64, 64, 128), (1, 4, 1))(x) + x,
        MeanPool((2, ))
    )
end

function model()
    return Chain(
        Conv((4, ), 1=>128, relu, pad=SamePad()),
        residual_block(),
        residual_block(),
        residual_block(),
        residual_block(),
        residual_block(),
        residual_block(),
        residual_block(),
        residual_block(),
        residual_block(),
        residual_block(),
        flatten,
        Dense(4*128, 2048),
        Dense(2048, 2*dim*dim)
    )
end

file_names = readdir(SqState.training_data_path())
f = jldopen(joinpath(SqState.training_data_path(), file_names[1]), "r")
points = f["points"]
𝛒s = f["𝛒s"]

for i in 1:1 # 10000
    x = Float32.(points[:, i])
    y = ComplexF32.(𝛒s[i])

    x = reshape(x, (4096, 1, 1)) # 4096 points 1 channel, 1 data in a batch
    ŷ = reshape(model()(x), :)

    @show size(ŷ)
    @show loss(ŷ, y)
end
