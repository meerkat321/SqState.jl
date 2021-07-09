using SqState
using Flux
using CUDA

is_gpu = true

if CUDA.has_cuda() && is_gpu
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

dim = 70

file_names = readdir(SqState.training_data_path())
training_loader = preprocess(file_names[1], batch_size=100)

function conv_layers(ch::NTuple{4, <:Integer}, kernel_size::NTuple{3, <:Integer}, pad::NTuple{3, <:Any})
    return Chain(
        Conv((kernel_size[1], ), ch[1]=>ch[2], pad=pad[1]),
        BatchNorm(ch[2], relu),
        Conv((kernel_size[2], ), ch[2]=>ch[3], pad=pad[2]),
        BatchNorm(ch[3], relu),
        Conv((kernel_size[3], ), ch[3]=>ch[4], pad=pad[3]),
        BatchNorm(ch[4]),
    )
end

function conv_short_cut(ch::NTuple{2, <:Integer}, kernel_size::Integer, pad::Any)
    return Chain(
        Conv((kernel_size, ), ch[1]=>ch[2], pad=pad),
        BatchNorm(ch[2])
    )
end

function model()
    return Chain(
        # stage 0
        BatchNorm(1),
        Conv((31, ), 1=>64, pad=15),
        BatchNorm(64, relu),
        # res 1
        Parallel(+,
            conv_layers((64, 32, 32, 96), (1, 15, 7), (0, 7, 3)),
            conv_short_cut((64, 96), 1, 0),
        ),
        x -> relu.(x),
        MeanPool((2, )),
        BatchNorm(96, relu),
        # res 2
        SkipConnection(conv_layers((96, 32, 32, 96), (1, 7, 1), (0, 3, 0)), +),
        x -> relu.(x),
        # res 3
        SkipConnection(conv_layers((96, 32, 32, 96), (1, 7, 1), (0, 3, 0)), +),
        x -> relu.(x),
        MeanPool((4, )),
        BatchNorm(96, relu),
        # res 4
        Parallel(+,
            conv_layers((96, 64, 64, 128), (1, 3, 1), (0, 1, 0)),
            conv_short_cut((96, 128), 1, 0),
        ),
        x -> relu.(x),
    )
end

m = is_gpu ? model() |> gpu : model()
ps = Flux.params(m)
opt = ADAM(1e-4, (0.7, 0.9))

loss(x, y) = Flux.mse(m(x), y)

# for e in 1:10
#     l = 0f0
    for (i, (x, y)) in enumerate(training_loader)
        x = is_gpu ? x |> gpu : x
        y = is_gpu ? y |> gpu : y
        # @info "batch: $i"
        # @show size(x)
        # @show size(y)
        @show size(m(x))
        # @show loss(x, y)
        # gs = Flux.gradient(() -> loss(x, y), ps)
        # Flux.update!(opt, ps, gs)

        # l = loss(x, y)
        break
    end
#     @show l
# end
