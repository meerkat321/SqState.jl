using SqState
using Flux

dir = "sqth"
device = SqState.get_device()
model_name = "q2args_sqth_cnn"
batch_size = 25
epochs = 30
η₀ = 1e-2

m = SqState.cnn_q2args_sqth() |> device
loss(𝐱, 𝐲) = Flux.mse(m(𝐱), 𝐲)
opt = Flux.ADAM(η₀)

# prepare data
data_files = readdir(joinpath(SqState.training_data_path(), dir), join=true)
loader4test = SqState.preprocess_q2args(data_files[1], batch_size=batch_size)
@info "numbers of data fragments: $(length(data_files)-1)"
data_loaders = Channel(5, spawn=true) do ch
    for e in 1:epochs, (i, data_file) in enumerate(data_files[2:end])
        put!(ch, SqState.preprocess_q2args(data_file, batch_size=batch_size))
        @info "Load epoch $e, file $i into buffer"
        flush(stdout)
    end
end

t = Ref{Int64}(1)
losses = Float32[]
data4validation = [(reshape(𝐱[1, :, :], 4096, 1, :), 𝐲) for (𝐱, 𝐲) in loader4test] |> device
for loader4train in data_loaders
    @time begin
        data = [(reshape(𝐱[1, :, :], 4096, 1, :), 𝐲) for (𝐱, 𝐲) in loader4train] |> device

        # descent η
        (t[] > 100) && (opt.eta = η₀ / 2^ceil((t[]-100)/100))

        # training
        Flux.train!(loss, params(m), data, opt)

        # collect loss
        validation_loss = sum(loss(𝐱, 𝐲) for (𝐱, 𝐲) in data4validation)/length(data4validation)
        in_data_loss = sum(loss(𝐱, 𝐲) for (𝐱, 𝐲) in data)/length(data)
        @info "$(t[])0k data\n η: $(opt.eta)\n in  loss: $in_data_loss\n out loss: $validation_loss"

        # update saved model
        push!(losses, validation_loss)
        (losses[end] == minimum(losses)) && SqState.update_model!(SqState.model_path(), model_name, m)

        # update indicator
        t[] += 1
    end
end
