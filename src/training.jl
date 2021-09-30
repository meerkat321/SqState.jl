export
    train_q2ρ_sqth_th,
    train_q2argv_sqth,
    train_q2argv_sqth_th

function update_model!(model_path::String, model_name::String, model)
    model = cpu(model)
    jldsave(joinpath(model_path, "$model_name.jld2"); model)
    @warn "'$model_name' updated!"
end

function train_q2ρ_sqth_th(model_name::String; epochs=1, η₀=1e-5, batch_size=25)
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    prefix = "sqth_th"

    m = model_q2ρ_sqth_th() |> device
    loss(𝐱, 𝐲) = Flux.mse(m(𝐱), 𝐲)
    opt = Flux.Optimiser(WeightDecay(1e-4), Flux.ADAM(η₀))

    # prepare data
    data_file_names = filter(x->x!=".gitkeep", readdir(joinpath(SqState.training_data_path(), prefix)))
    loader_test = preprocess_q2ρ(prefix, data_file_names[1], batch_size=batch_size)
    @info "numbers of data fragments: $(length(data_file_names)-1)"
    data_loaders = Channel(5, spawn=true) do ch
        for e in 1:epochs, (i, file_name) in enumerate(data_file_names[2:end])
            put!(ch, preprocess_q2ρ(prefix, file_name, batch_size=batch_size))
            @info "Load epoch $e, file $i into buffer"
            flush(stdout)
        end
    end

    t = 1
    losses = Float32[]
    data_validation = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_test] |> device
    for loader_train in data_loaders
        @time begin
            data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device

            # descent η
            (t > 20) && (opt.os[2].eta = η₀ / 2^ceil((t-20)/20))

            # training
            Flux.train!(loss, params(m), data, opt)

            # collect loss
            validation_loss = sum(loss(𝐱, 𝐲) for (𝐱, 𝐲) in data_validation)/length(data_validation)
            in_data_loss = sum(loss(𝐱, 𝐲) for (𝐱, 𝐲) in data)/length(data)
            @info "$(t)0k data\n η: $(opt.os[2].eta)\n in  loss: $in_data_loss\n out loss: $validation_loss"

            # update saved model
            push!(losses, validation_loss)
            (losses[end] == minimum(losses)) && update_model!(model_path(), model_name, m)

            # update indicator
            t += 1
        end
    end
end

function train_q2argv_sqth(model_name::String; epochs=10, η₀=1e-3, batch_size=25)
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    prefix = "sqth"

    m = model_q2args_sqth() |> device
    loss(𝐱, 𝐲) = Flux.mse(m(𝐱), 𝐲)
    opt = Flux.Optimiser(WeightDecay(1e-4), Flux.ADAM(η₀))

    # prepare data
    data_file_names = filter(x->x!=".gitkeep", readdir(joinpath(SqState.training_data_path(), prefix)))
    loader_test = preprocess_q2args(prefix, data_file_names[1], batch_size=batch_size)
    @info "numbers of data fragments: $(length(data_file_names)-1)"
    data_loaders = Channel(5, spawn=true) do ch
        for e in 1:epochs, (i, file_name) in enumerate(data_file_names[2:end])
            put!(ch, preprocess_q2args(prefix, file_name, batch_size=batch_size))
            @info "Load epoch $e, file $i into buffer"
            flush(stdout)
        end
    end

    t = 1
    losses = Float32[]
    data_validation = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_test] |> device
    for loader_train in data_loaders
        @time begin
            data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device

            # descent η
            (t > 100) && (opt.os[2].eta = η₀ / 2^ceil((t-100)/100))

            # training
            Flux.train!(loss, params(m), data, opt)

            # collect loss
            validation_loss = sum(loss(𝐱, 𝐲) for (𝐱, 𝐲) in data_validation)/length(data_validation)
            in_data_loss = sum(loss(𝐱, 𝐲) for (𝐱, 𝐲) in data)/length(data)
            @info "$(t)0k data\n η: $(opt.os[2].eta)\n in  loss: $in_data_loss\n out loss: $validation_loss"

            # update saved model
            push!(losses, validation_loss)
            (losses[end] == minimum(losses)) && update_model!(model_path(), model_name, m)

            # update indicator
            t += 1
        end
    end
end

function train_q2argv_sqth_th(model_name::String; epochs=10, η₀=1e-3, batch_size=25)
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    prefix = "sqth_th"

    m = model_q2args_sqth_th() |> device
    loss(𝐱, 𝐲) = Flux.mse(m(𝐱), 𝐲)
    opt = Flux.Optimiser(WeightDecay(1e-4), Flux.ADAM(η₀))

    # prepare data
    data_file_names = filter(x->x!=".gitkeep", readdir(joinpath(SqState.training_data_path(), prefix)))
    loader_test = preprocess_q2args(prefix, data_file_names[1], batch_size=batch_size)
    @info "numbers of data fragments: $(length(data_file_names)-1)"
    data_loaders = Channel(5, spawn=true) do ch
        for e in 1:epochs, (i, file_name) in enumerate(data_file_names[2:end])
            put!(ch, preprocess_q2args(prefix, file_name, batch_size=batch_size))
            @info "Load epoch $e, file $i into buffer"
            flush(stdout)
        end
    end

    t = 1
    losses = Float32[]
    data_validation = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_test] |> device
    for loader_train in data_loaders
        @time begin
            data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device

            # descent η
            (t > 200) && (opt.os[2].eta = η₀ / 2^ceil((t-200)/200))

            # training
            Flux.train!(loss, params(m), data, opt)

            # collect loss
            validation_loss = sum(loss(𝐱, 𝐲) for (𝐱, 𝐲) in data_validation)/length(data_validation)
            in_data_loss = sum(loss(𝐱, 𝐲) for (𝐱, 𝐲) in data)/length(data)
            @info "$(t)0k data\n η: $(opt.os[2].eta)\n in  loss: $in_data_loss\n out loss: $validation_loss"

            # update saved model
            push!(losses, validation_loss)
            (losses[end] == minimum(losses)) && update_model!(model_path(), model_name, m)

            # update indicator
            t += 1
        end
    end
end
