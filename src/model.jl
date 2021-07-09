using Flux
using CUDA

# if CUDA.has_cuda()
#     @info "CUDA is on"
#     CUDA.allowscalar(true)
# end

file_names = readdir(SqState.training_data_path())
training_loader = preprocess(file_names[1], batch_size)

# vanilla_softplus(x) = log1p.(exp.(x))

# c_glorot_uniform(dims...) = Flux.glorot_uniform(dims...) + Flux.glorot_uniform(dims...) * im

# function C_BatchNorm(
#     chs::Int,
#     λ=identity;
#     initβ=i->zeros(ComplexF32, i),
#     initγ=i->ones(ComplexF32, i),
#     affine=true,
#     track_stats=true,
#     ϵ=1f-5+1f-5im,
#     momentum=1f-1+1f-1im
# )
#     β = affine ? initβ(chs) : nothing
#     γ = affine ? initγ(chs) : nothing
#     μ = track_stats ? zeros(ComplexF32, chs) : nothing
#     σ² = track_stats ? ones(ComplexF32, chs) : nothing

#     return Flux.BatchNorm(λ, β, γ, μ, σ², ϵ, momentum, affine, track_stats, nothing, chs)
# end

# function conv_layers(ch::NTuple{4, <:Integer}, kernel_size::NTuple{3, <:Integer}, pad::NTuple{3, <:Integer})
#     return Chain(
#         Conv((kernel_size[1], ), ch[1]=>ch[2], pad=pad[1]),
#         C_BatchNorm(ch[2], vanilla_softplus),
#         Conv((kernel_size[2], ), ch[2]=>ch[3], pad=pad[2]),
#         C_BatchNorm(ch[3], vanilla_softplus),
#         Conv((kernel_size[3], ), ch[3]=>ch[4], pad=pad[3]),
#         C_BatchNorm(ch[4], vanilla_softplus),
#     )
# end

# function residual_block()
#     return Chain(
#         SkipConnection(conv_layers((128, 64, 64, 128), (1, 5, 1), (0, 2, 0)), +),
#         MeanPool((2, ))
#     )
# end

# function model()
#     return Chain(
#         Conv((5, ), 1=>128, vanilla_softplus, pad=2),
#         Chain([residual_block() for _ = 1:10]...),
#         flatten,
#         Dense(4*128, 2048),
#         Dense(2048, dim*dim)
#     )
# end

# m = model() |> gpu
# batchsize = 2

# function loss(x, 𝐲)
#     𝐥̂ = reshape(m(x), (dim, dim, batchsize))
#     l = sum(sqrt(abs(Flux.mse(𝐥̂[:, :, i]' * 𝐥̂[:, :, i], 𝐲[:, :, i]))) for i in 1:batchsize) / batchsize

#     return l
# end

# opt = Momentum(1e-10)
# ps = Flux.params(m)

# for (i, (x, y)) in enumerate(train_loader)
#     x, y = x|>gpu, y|>gpu
#     @info "batch: $i"
#     @show size(x)
#     @show size(y)
#     @show loss(x, y)
#     gs = Flux.gradient(() -> loss(x, 𝐲), ps)
#     Flux.update!(opt, ps, gs)
#     break
# end
