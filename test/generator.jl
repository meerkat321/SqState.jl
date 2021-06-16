using SqState
using JLD2
using DataDeps
using CUDA
using Flux
using StatsBase

function fetch_data()
    data_path = joinpath(datadep"SqState", "training_data", "gen_data")
    file_names = readdir(data_path)
    𝐩_dict = jldopen(joinpath(data_path, file_names[1]), "r")["𝐩_dict"]

    return 𝐩_dict
end

function construct_model()
    return Chain(
        Dense(3, 16, σ=relu),
        Dense(16, 64, σ=relu),
        Dense(64, 256, σ=relu),
        Dense(256, 1024, σ=relu),
        Dense(1024, 4096),
        Dense(4096, 16384),
        Dense(4096, 32768)
    )
end

function loss(generated_data, 𝐩)
    generated_data = reshape(generated_data, Int(length(generated_data)/2), 2)
    𝐩̂ = fit(Histogram, (generated_data[:, 1], generated_data[:, 2])).weights

    return crossentropy(𝐩̂, 𝐩, agg=mean)
end

function preprocess(data::Dict)
    xs = hcat([[k...] for (k, _) in data]...)
    ys = [v for (_, v) in data]

    return xs, ys
end

function main()
    data = fetch_data()
    train_loader = Flux.Data.DataLoader(preprocess(data), batchsize=20)

    for epoch in 1:100
        for (x, y) in train_loader
            @assert size(x) == (3, 20)
            @assert size(y) == (20,)
        end
    end
end
