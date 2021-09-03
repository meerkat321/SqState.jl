export
    train,
    get_model

function update_model!(model_file_path, model)
    model = cpu(model)
    jldsave(model_file_path; model)
    @warn "model updated!"
end

function train(model_name::String; epochs=5, η₀=1f-2, batch_size=100)
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    m = model() |> device
    loss(x, y) = Flux.mse(m(x), y)
    opt = Flux.ADAM(η₀)

    # prepare data
    data_file_names = filter(x->x!=".gitkeep", readdir(SqState.training_data_path()))
    loader_test = preprocess(data_file_names[1], batch_size=batch_size)
    @info "numbers of data fragments: $(length(data_file_names)-1)"
    data_loaders = Channel(5, spawn=true) do ch
        for e in 1:epochs, (i, file_name) in enumerate(data_file_names[2:end])
            put!(ch, preprocess(file_name, batch_size=batch_size))
            @info "Load epoch $e, file $i into buffer"
        end
    end

    t = 1
    losses = Float32[]
    function validate()
        validation_loss = sum(loss(device(𝐱), device(𝐲)) for (𝐱, 𝐲) in loader_test)/length(loader_test)
        @info "$(t)0k data\n η: $(opt.eta)\n loss: $validation_loss"

        push!(losses, validation_loss)
        (losses[end] == minimum(losses)) && update_model!(joinpath(model_path(), "$model_name.jld2"), m)
    end
    call_back = Flux.throttle(validate, 10, leading=false, trailing=true)

    for e in 1:epochs
        @info "Epoch $e"
        for loader_train in data_loaders
            data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device

            @time Flux.train!(loss, params(m), data, opt, cb=call_back)
            (t % 50 == 0) && (opt.eta /= 2)
            t += 1
        end
    end
end

function get_model(model_name::String)
    f = jldopen(joinpath(model_path() , "$model_name.jld2"))
    model = f["model"]
    close(f)

    return model
end
