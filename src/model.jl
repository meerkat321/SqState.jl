export
    training_process,
    get_model

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

function short_cut(ch::NTuple{2, <:Integer}, kernel_size::Integer, pad::Any)
    return Chain(
        Conv((kernel_size, ), ch[1]=>ch[2], pad=pad),
        BatchNorm(ch[2])
    )
end

function model(; dim=70)
    return Chain(
        # stage 0
        BatchNorm(1),
        Conv((31, ), 1=>64, pad=15),
        BatchNorm(64, relu),
        # res 1
        Parallel(+,
            conv_layers((64, 32, 32, 96), (1, 15, 7), (0, 7, 3)),
            short_cut((64, 96), 1, 0),
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
            short_cut((96, 128), 1, 0),
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
            short_cut((128, 196), 1, 0),
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
        Dense(32*196, Int32(dim*dim/4), relu),
        Dense(Int32(dim*dim/4), dim*dim, tanh),
        x -> x ./ sqrt(max(sum(x.^2), 1f-12)) # l-2 normalize
    )
end

function training_process(model_name;
    file_names=readdir(SqState.training_data_path()),
    batch_size=100, n_batch=60, epochs=10,
    is_gpu=true
)
    model_file_path = joinpath(mkpath(model_path()), "$model_name.jld2")
    if CUDA.has_cuda() && is_gpu
        @info "CUDA is on"
        CUDA.allowscalar(false)
    end

    # prepare model
    m = is_gpu ? model() |> gpu : model()
    loss(x, y) = Flux.mse(m(x), y)
    ps = Flux.params(m)
    opt = ADAM(1e-2, (0.7, 0.9))

    # jit model
    @time begin
        @info "jit..."
        x, y = first(preprocess(file_names[2], batch_size=1))
        x = is_gpu ? x |> gpu : x
        y = is_gpu ? y |> gpu : y
        gs = Flux.gradient(() -> loss(x, y), ps)
        Flux.update!(opt, ps, gs)
    end

    # prepare data
    test_data_loader = preprocess(file_names[1], batch_size=batch_size)
    @info "numbers of data fragments: $n_batch/$(length(file_names)-1)"
    data_loaders = Channel(5, spawn=true) do ch
        for e in 1:epochs
            for (i, file_name) in enumerate(file_names[2:(n_batch+1)])
                put!(ch, preprocess(file_name, batch_size=batch_size))
                @info "Load epoch $(e), $(i)th files into buffer"
            end
        end
    end

    # training
    in_losses = Float32[]
    out_losses = Float32[]
    for (t, loader) in enumerate(data_loaders)
        in_loss = out_loss = 0
        bs = length(loader)

        t1 = time()
        for (b, (x, y)) in enumerate(loader)
            x = is_gpu ? x |> gpu : x
            y = is_gpu ? y |> gpu : y

            # (t ≥ 20) && (opt.eta > 1e-7) && (opt.eta = 1e-2 / 2^((length(loader)*(t-15)+b)/(5*length(loader))))
            (t ≥ 20) && ((t*bs+b) % (5*bs) == 0) && (opt.eta > 1e-4) && (opt.eta /= 2)

            gs = Flux.gradient(() -> loss(x, y), ps)
            Flux.update!(opt, ps, gs)
            in_loss += loss(x, y)
            out_loss += validation(test_data_loader, loss, is_gpu)
        end

        push!(in_losses, in_loss/bs)
        push!(out_losses, out_loss/bs)
        moniter(t*length(loader), t1, opt, bs, in_losses, out_losses)
        (out_losses[end] == minimum(out_losses)) && (update_model!(model_file_path, model_name, m, in_losses, out_losses))
    end

    return m, in_losses, out_losses
end

function validation(test_data_loader::DataLoader, loss_func, is_gpu)
    x, y = first(test_data_loader)
    x = is_gpu ? x |> gpu : x
    y = is_gpu ? y |> gpu : y

    return loss_func(x, y)
end

function update_model!(model_file_path, model_name, model, in_losses, out_losses)
    model = cpu(model) #return model to cpu before serialization
    jldsave(model_file_path; model, in_losses, out_losses)
    @warn "'$model_name' model updated!"
end

function moniter(t, t1, opt, n, in_losses, out_losses)
    plt = scatterplot(in_losses, xlabel="batchs/$n", name="In", width=100, color=:green)
    plt = scatterplot!(plt, out_losses, name="Out", color=:red)

    print("\e[H\e[2J")
    println(plt)
    @info "$t\n" *
        "# time: $(time()-t1)\n" *
        "# learning rate: $(opt.eta)\n" *
        "# in data loss:  $(in_losses[end])\n" *
        "# out data loss: $(out_losses[end])\n"
end

function get_model(model_name::String)
    return jldopen(joinpath(model_path() , "$model_name.jld2"))["model"]
end
