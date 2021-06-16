using SqState
using JLD2
using DataDeps
using CUDA
using Flux
using IterTools
using StatsBase

function fetch_data()
    data_path = joinpath(datadep"SqState", "training_data", "gen_data")
    file_names = readdir(data_path)
    𝐩_dict = jldopen(joinpath(data_path, file_names[1]), "r")["𝐩_dict"]

    return 𝐩_dict
end

function construct_model()
    return Chain(
        Dense(3, 16, relu),
        Dense(16, 64, relu),
        Dense(64, 256, relu),
        Dense(256, 1024, relu),
        Dense(1024, 4096),
        Dense(4096, 16384),
        Dense(16384, 32768)
    )
end

function sq_loss(model, args, 𝐩)
    generated_data = model(args)
    generated_data = reshape(generated_data, Int(length(generated_data)/2), 2)
    h = fit(Histogram, (generated_data[:, 1], generated_data[:, 2]), nbins=40).weights
    𝐩̂ = h / sum(h)

    return crossentropy(𝐩̂, 𝐩, agg=mean)
end

function preprocess(data::Dict)
    xs = hcat([[k...] for (k, _) in data]...)
    ys = [v for (_, v) in data]

    return xs, ys
end

function main()
    data = fetch_data()
    train_loader = Flux.Data.DataLoader(preprocess(data), batchsize=20, shuffle=true)

    model = construct_model()
    loss(x, y) = sq_loss(model, x, y)
    ps = Flux.params(model)

    loss(first(train_loader)[1][:, 1], first(train_loader)[2][1])

    # for epoch in 1:100
    #     for batch_data in train_loader
    #         # @assert size(batch_data[1]) == (3, 20)
    #         # @assert size(batch_data[2]) == (20,)
    #         # Flux.train!(loss, ps, batch_data, ADAM())
    #     end
    # end

    # Flux.train!(loss, ps, ncycle(train_loader, 10), ADAM())
end

# TODO: get nbins from data
