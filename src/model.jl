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

function shortcut(ch::NTuple{2, <:Integer}, kernel_size::Integer, pad::Any)
    return Chain(
        Conv((kernel_size, ), ch[1]=>ch[2], pad=pad),
        BatchNorm(ch[2])
    )
end

function res_block(
    conv_ch::NTuple{4, <:Integer},
    conv_kernel_size::NTuple{3, <:Integer},
    conv_pad::NTuple{3, <:Any},
    shortcut_kernel_size::Integer,
    shortcut_pad::Any,
    pool_size::Integer;
)
    pool = (pool_size > 0) ? MeanPool((pool_size, )) : identity

    return Chain(
        Parallel(+,
            conv_layers(conv_ch, conv_kernel_size, conv_pad),
            shortcut((conv_ch[1], conv_ch[end]), shortcut_kernel_size, shortcut_pad),
        ),
        x -> relu.(x),
        pool,
        BatchNorm(conv_ch[end], relu)
    )
end

function model()
    return Chain(
        # stage 0
        BatchNorm(1),
        Conv((31, ), 1=>8, pad=15),
        BatchNorm(8, relu),

        # res
        res_block((8, 4, 4, 16), (1, 15, 7), (0, 7, 3), 1, 0, -1),
        res_block((16, 8, 8, 32), (1, 15, 7), (0, 7, 3), 1, 0, -1),
        res_block((32, 16, 16, 64), (1, 15, 7), (0, 7, 3), 1, 0, -1),
        res_block((64, 32, 32, 128), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((128, 32, 32, 64), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((64, 16, 16, 32), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((32, 8, 8, 16), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((16, 4, 4, 8), (1, 15, 7), (0, 7, 3), 1, 0, 2),

        # stage 1
        flatten,
        Dense(8*128, 64, relu),
        Dense(64, 16, relu),
        Dense(16, 3, relu)
    )
end

function training_process(
    model_name;
    data_file_names=readdir(SqState.training_data_path()),
    batch_size=100, n_batch=99, epochs=3,
)
    model_file_path = joinpath(mkpath(model_path()), "$model_name.jld2")
    if CUDA.has_cuda()
        @info "CUDA is on"
        CUDA.allowscalar(false)
    else
        throw("No Nvidia gpu")
    end

    # prepare model
    m = gpu(model())
    loss(x, y) = Flux.mse(m(x), y)
    ps = Flux.params(m)
    opt = ADAM(1e-2)

    # jit model
    @time begin
        @info "jit..."
        x, y = first(preprocess(data_file_names[2], batch_size=1))
        x, y = gpu(x), gpu(y)
        gs = Flux.gradient(() -> loss(x, y), ps)
        Flux.update!(opt, ps, gs)
    end

    # prepare data
    test_data_loader = preprocess(data_file_names[1], batch_size=batch_size)
    @info "numbers of data fragments: $n_batch/$(length(data_file_names)-1)"
    data_loaders = Channel(5, spawn=true) do ch
        for e in 1:epochs
            for (i, file_name) in enumerate(data_file_names[2:(n_batch+1)])
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

        t_threshold = 30
        if t > t_threshold
            opt.eta = 1e-2 / 2^ceil((t-t_threshold)/30)
        end

        t1 = time()
        for (x, y) in loader
            x, y = gpu(x), gpu(y)
            gs = Flux.gradient(() -> loss(x, y), ps)
            Flux.update!(opt, ps, gs)

            in_loss += loss(x, y)
            out_loss += validation(test_data_loader, loss)
        end

        push!(in_losses, in_loss/bs)
        push!(out_losses, out_loss/bs)
        (t > 1) && moniter(t, t1, opt, bs, in_losses, out_losses)
        (out_losses[end] == minimum(out_losses)) && (update_model!(model_file_path, model_name, m, in_losses, out_losses))
    end

    return m, in_losses, out_losses
end

function validation(test_data_loader::DataLoader, loss_func)
    x, y = first(test_data_loader)
    x, y = gpu(x), gpu(y)

    return loss_func(x, y)
end

function update_model!(model_file_path, model_name, model, in_losses, out_losses)
    model = cpu(model)
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
        "# out data loss: $(out_losses[end])\n" *
        "# Δloss: $(out_losses[end] - out_losses[end-1])\n"
end

function get_model(model_name::String)
    return jldopen(joinpath(model_path() , "$model_name.jld2"))["model"]
end
