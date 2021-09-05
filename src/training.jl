export
    train,
    get_model

function update_model!(model_file_path, model)
    model = cpu(model)
    jldsave(model_file_path; model)
    @warn "model updated!"
end

function train(model_name::String; epochs=10, η₀=1e-2, batch_size=25, n_validation_batch=100)
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    m = model() |> device
    loss(x, y) = Flux.mse(m(x), y)
    hloss(x, y) = Flux.huber_loss(m(x), y)
    opt = Flux.Momentum(η₀, 0.9)

    # prepare data
    data_file_names = filter(x->x!=".gitkeep", readdir(SqState.training_data_path()))
    loader_test = preprocess_q2args(data_file_names[1], batch_size=batch_size)
    @info "numbers of data fragments: $(length(data_file_names)-1)"
    data_loaders = Channel(5, spawn=true) do ch
        for e in 1:epochs, (i, file_name) in enumerate(data_file_names[2:end])
            put!(ch, preprocess_q2args(file_name, batch_size=batch_size))
            @info "Load epoch $e, file $i into buffer"
        end
    end

    t = 1
    losses = Float32[]
    data_validation = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_test][1:n_validation_batch] |> device
    function validate()
        validation_loss = sum(hloss(𝐱, 𝐲) for (𝐱, 𝐲) in data_validation)/length(data_validation)
        @info "$(t)0k data\n η: $(opt.eta)\n loss: $validation_loss"

        push!(losses, validation_loss)
        (losses[end] == minimum(losses)) && update_model!(joinpath(model_path(), "$model_name.jld2"), m)
    end
    call_back = Flux.throttle(validate, 20, leading=false, trailing=true)

    for loader_train in data_loaders
        data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device
        @time Flux.train!(loss, params(m), data, opt, cb=call_back)
        (t % 30 == 0) && (opt.eta /= 2)
        t += 1
    end
end

function get_model(model_name::String)
    f = jldopen(joinpath(model_path() , "$model_name.jld2"))
    model = f["model"]
    close(f)

    return model
end
