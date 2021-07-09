using SqState
using Flux
using CUDA

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

dim = 70

file_names = readdir(SqState.training_data_path())
training_loader = preprocess(file_names[1], batch_size=100)

function conv_layers(ch::NTuple{4, <:Integer}, kernel_size::NTuple{3, <:Integer}, pad::NTuple{3, <:Integer})
    return Chain(
        Conv((kernel_size[1], ), ch[1]=>ch[2], pad=pad[1]),
        BatchNorm(ch[2], relu),
        Conv((kernel_size[2], ), ch[2]=>ch[3], pad=pad[2]),
        BatchNorm(ch[3], relu),
        Conv((kernel_size[3], ), ch[3]=>ch[4], pad=pad[3]),
        BatchNorm(ch[4], relu),
    )
end

function residual_block()
    return Chain(
        SkipConnection(conv_layers((128, 64, 64, 128), (1, 5, 1), (0, 2, 0)), +),
        MeanPool((2, ))
    )
end

function model()
    return Chain(
        Conv((5, ), 1=>128, relu, pad=2),
        Chain([residual_block() for _ = 1:10]...),
        flatten,
        Dense(4*128, 2048),
        Dense(2048, dim*dim)
        # Dense(4096*128, dim*dim)
    )
end

m = model() |> gpu
ps = Flux.params(m)
opt = ADAM(1e-4)

loss(x, y) = Flux.mse(m(x), y)

for e in 1:10
    l = 0f0
    for (i, (x, y)) in enumerate(training_loader)
        x, y = x|>gpu, y|>gpu
        # @info "batch: $i"
        # @show size(x)
        # @show size(y)
        # @show size(m(x))
        # @show loss(x, y)
        gs = Flux.gradient(() -> loss(x, y), ps)
        Flux.update!(opt, ps, gs)

        l = loss(x, y)
    end
    @show l
end
