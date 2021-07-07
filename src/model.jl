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

function loss(ŷ, 𝐲)
    l = sum(
        diagm((i-dim) => ŷ[(sum(1:(i-1))+1):sum(1:i)])
        for i in 1:dim
    )

    return Flux.mse(l * l', 𝐲)
end

function conv_layers()
    return Chain(
        Conv((4, ), 1=>128, relu, stride=1, pad=SamePad()),
        Conv((1, ), 128=>64, relu, stride=1, pad=SamePad()),
        Conv((4, ), 64=>64, relu, stride=1, pad=SamePad()),
        Conv((1, ), 64=>128, relu, stride=1, pad=SamePad())
    )
end

function residual_block()
    return Chain(
        x -> conv_layers()(x) + Conv((3, ), 1=>128, relu, stride=1, pad=1)(x),
        MeanPool((2, ))
    )
end

function model()
    return Chain(
        residual_block(),
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
    ŷ = model()(x)
    println(size(ŷ))

    # println(loss(ŷ, y))
end
