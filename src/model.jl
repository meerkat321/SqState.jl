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

vanilla_softplus(x) = log1p(exp(x))

c_glorot_uniform(dims...) = Flux.glorot_uniform(dims...) + Flux.glorot_uniform(dims...) * im

function C_BatchNorm(chs::Int, λ=identity;
    initβ = i -> zeros(ComplexF32, i),
    initγ = i -> ones(ComplexF32, i),
    affine=true, track_stats=true,
    ϵ=1f-5 + 1f-5im, momentum=1f-1 + 1f-1im
)

    β = affine ? initβ(chs) : nothing
    γ = affine ? initγ(chs) : nothing
    μ = track_stats ? zeros(ComplexF32, chs) : nothing
    σ² = track_stats ? ones(ComplexF32, chs) : nothing

    return Flux.BatchNorm(
        λ, β, γ,
        μ, σ², ϵ, momentum,
        affine, track_stats,
        nothing, chs
    )
end

function conv_layers(ch::NTuple{4, <:Integer}, kernel_size::NTuple{3, <:Integer})
    return [
        Conv((kernel_size[1], ), ch[1]=>ch[2], pad=SamePad(), init=c_glorot_uniform),
        C_BatchNorm(ch[2], vanilla_softplus),
        Conv((kernel_size[2], ), ch[2]=>ch[3], pad=SamePad(), init=c_glorot_uniform),
        C_BatchNorm(ch[3], vanilla_softplus),
        Conv((kernel_size[3], ), ch[3]=>ch[4], pad=SamePad(), init=c_glorot_uniform),
        C_BatchNorm(ch[4], vanilla_softplus),
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
        Conv((4, ), 1=>128, vanilla_softplus, pad=SamePad()),
        Chain(res_blk...),
        flatten,
        Dense(4*128, 2048),
        Dense(2048, dim*dim)
    )
end

m = model()

function loss(x, 𝐲)
    𝐥̂ = reshape(m(x), (dim, dim))
    𝛒̂ = 𝐥̂ * 𝐥̂'

    return abs(Flux.mse(𝛒̂, 𝐲))
end

file_names = readdir(SqState.training_data_path())
f = jldopen(joinpath(SqState.training_data_path(), file_names[1]), "r")
points = f["points"]
𝛒s = f["𝛒s"]

for i in 1:1 # 10000
    x = reshape(ComplexF32.(points[:, i]), (4096, 1, 1)) # 4096 points 1 channel, 1 data in a batch
    𝐲 = ComplexF32.(𝛒s[i])

    @show size(reshape(m(x), :))
    @show loss(x, 𝐲)
    @show gradient(x->sum(abs, m(x)), x)
    @show gradient(x->loss(x, 𝐲), x)
end
