export
    training_process,
    get_model

function validation(data_loader::DataLoader, loss_func)
    losses = 0f0
    for (x, y) in data_loader
        losses += loss_func(gpu(x), gpu(y))
    end

    return losses / length(data_loader)
end

function update_model!(model_file_path, model_name, model, in_losses, out_losses)
    model = cpu(model)
    jldsave(model_file_path; model, in_losses, out_losses)
    @warn "'$model_name' model updated!"
end

function moniter(f, η, in_losses, out_losses, Δt)
    plt = scatterplot(in_losses, xlabel="10K", name="In", width=90, color=:green)
    plt = scatterplot!(plt, out_losses, name="Out", color=:red)

    d, h, m, s = format_time(Δt)
    info_str = "$f 10K\n" *
        "# time: $d $h $m $s\n" *
        "# learning rate: $η\n" *
        "# in data loss:  $(in_losses[end])\n" *
        "# out data loss: $(out_losses[end])\n"
    (f > 1) && (info_str *= "# Δloss: $(out_losses[end] - out_losses[end-1])\n")

    print("\e[H\e[2J")
    println(plt)
    @info info_str
end

function format_time(Δt::Millisecond)
    d = floor(Δt, Day)
    Δt -= Millisecond(d)
    h = floor(Δt, Hour)
    Δt -= Millisecond(h)
    m = floor(Δt, Minute)
    Δt -= Millisecond(m)
    s = floor(Δt, Second)

    return d, h, m, s
end

function training_process(
    model_name;
    data_file_names=readdir(SqState.training_data_path()),
    batch_size=100, n_batch=100, epochs=6,
    η₀=1e-2, f_threshold=100, Δf=100,
    show_moniter=true
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

    # loss and opt
    in_losses = Float32[]
    out_losses = Float32[]
    loss(x, y) = Flux.mse(m(x), y)
    opt = ADAM(η₀)

    # jit model
    @time begin
        @info "jit..."
        x, y = first(preprocess(data_file_names[2], batch_size=1))
        Flux.train!(loss, Flux.params(m), [(gpu(x), gpu(y))], opt)
    end

    # prepare data
    (".gitkeep" in data_file_names) && (data_file_names = filter(x->x!=".gitkeep", data_file_names))
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
    t0 = now()
    for (f, loader) in enumerate(data_loaders)
        data = [(gpu(x), gpu(y)) for (x, y) in loader]

        # adjust learning rate
        (f > f_threshold) && (opt.eta = η₀ / 2^ceil((f-f_threshold)/Δf))

        Flux.train!(loss, Flux.params(m), data, opt)

        # trace loss and update stored model
        push!(in_losses, validation(loader, loss))
        push!(out_losses, validation(test_data_loader, loss))
        show_moniter && moniter(f, opt.eta, in_losses, out_losses, now()-t0)
        (out_losses[end] == minimum(out_losses)) && (update_model!(model_file_path, model_name, m, in_losses, out_losses))
    end

    return m, in_losses, out_losses
end

function get_model(model_name::String)
    f = jldopen(joinpath(model_path() , "$model_name.jld2"))
    model = f["model"]
    close(f)

    return model
end
