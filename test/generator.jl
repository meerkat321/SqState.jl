using SqState
using JLD2
using DataDeps
using CUDA
using Flux
using IterTools
using StatsBase

push_one_more!(v::Vector) = push!(v, v[end] + (v[end]-v[end-1]))

function fetch_data()
    data_path = joinpath(datadep"SqState", "training_data", "gen_data")
    file_names = readdir(data_path)
    f = jldopen(joinpath(data_path, file_names[1]), "r")

    𝐩_dict = f["𝐩_dict"]
    bin_θs = [f["bin_θs"]...]
    push_one_more!(bin_θs)
    bin_xs = [f["bin_xs"]...]
    push_one_more!(bin_xs)

    return 𝐩_dict, bin_θs, bin_xs
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

function sq_loss(model, args, 𝐩, bin_θs, bin_xs)
    generated_data = model(args)
    generated_data = reshape(generated_data, Int(length(generated_data)/2), 2)

    h = fit(Histogram, (generated_data[:, 1], generated_data[:, 2]), (bin_θs, bin_xs)).weights
    𝐩̂ = h / sum(h)

    return crossentropy(𝐩̂, 𝐩)
end

function preprocess(data::Dict)
    xs = hcat([[k...] for (k, _) in data]...)
    ys = [v for (_, v) in data]

    return xs, ys
end

function main()
    data, bin_θs, bin_xs = fetch_data()
    train_loader = Flux.Data.DataLoader(preprocess(data), batchsize=20, shuffle=true)

    model = construct_model()
    loss(x, y) = sq_loss(model, x, y, bin_θs, bin_xs)

    
end

# TODO: get nbins from data
