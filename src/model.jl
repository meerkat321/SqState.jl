using Flux: length
using CUDA: length
using SqState
using Flux
using Flux.Data: DataLoader
using CUDA

is_gpu = true

if CUDA.has_cuda() && is_gpu
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

dim = 70

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
        # res 5
        SkipConnection(conv_layers((128, 64, 64, 128), (1, 3, 1), (0, 1, 0)), +),
        x -> relu.(x),
        # res 6
        SkipConnection(conv_layers((128, 64, 64, 128), (1, 3, 1), (0, 1, 0)), +),
        x -> relu.(x),
        MeanPool((8, )),
        BatchNorm(128, relu),
        # res 7
        Parallel(+,
            conv_layers((128, 96, 96, 196), (1, 3, 1), (0, 1, 0)),
            conv_short_cut((128, 196), 1, 0),
        ),
        x -> relu.(x),
        # res 8
        SkipConnection(conv_layers((196, 96, 96, 196), (1, 3, 1), (0, 1, 0)), +),
        x -> relu.(x),
        # res 9
        SkipConnection(conv_layers((196, 96, 96, 196), (1, 3, 1), (0, 1, 0)), +),
        x -> relu.(x),
        # res 10
        SkipConnection(conv_layers((196, 96, 96, 196), (1, 3, 1), (0, 1, 0)), +),
        x -> relu.(x),
        # res 11
        SkipConnection(conv_layers((196, 96, 96, 196), (1, 3, 1), (0, 1, 0)), +),
        x -> relu.(x),
        MeanPool((2, )),
        BatchNorm(196, relu),
        # stage 1
        flatten,
        Dense(32*196, 5586, relu),
        Dense(5586, dim*dim),
        x -> x ./ sum(x.^2) # l-2 normalize
    )
end

m = is_gpu ? model() |> gpu : model()
ps = Flux.params(m)
opt = ADAM(1e-2, (0.7, 0.9))

loss(x, y) = Flux.mse(m(x), y)

file_names = readdir(SqState.training_data_path())

function warm_up_jit()
    @info "jit..."
    x, y = first(preprocess(file_names[end], batch_size=1))
    x = is_gpu ? x |> gpu : x
    y = is_gpu ? y |> gpu : y
    gs = Flux.gradient(() -> loss(x, y), ps)
    Flux.update!(opt, ps, gs)
end

@time warm_up_jit()

@info "load data"
data_fragments = Vector{DataLoader}(undef, length(file_names))
@time @sync for (f, file_name) in enumerate(file_names)
    Threads.@spawn data_fragments[f] = preprocess(file_name, batch_size=100)
end

for e in 1:1
    @info "epoch: $e"
    for (f, loader) in enumerate(data_fragments[1:(end-1)])
        l = 0f0
        @time for (b, (x, y)) in enumerate(loader)
            x = is_gpu ? x |> gpu : x
            y = is_gpu ? y |> gpu : y

            gs = Flux.gradient(() -> loss(x, y), ps)
            Flux.update!(opt, ps, gs)

            l += loss(x, y)
        end
        @info "loss $f: $(l/100)"
    end
end

testing_loader = data_fragments[end]
test_loss = 0f0
for (x, y) in testing_loader
    x = is_gpu ? x |> gpu : x
    y = is_gpu ? y |> gpu : y
    test_loss += loss(x, y)
end
@info "Out data loss: $(test_loss/100)"
