using SqState
using JLD2
using LinearAlgebra
using Flux.Data: DataLoader

export
    𝛒2y,
    preprocess

function 𝛒2y(𝛒::Matrix; δ=1e-15)
    dim = size(𝛒, 1)
    𝛅 = Matrix{Float64}(I, dim, dim) * δ

    𝐥 = cholesky(Hermitian(𝛒 + 𝛅)).L
    l = vcat([diag(𝐥, i-dim) for i in 1:dim]...)

    return Float32.(vcat(real.(l), imag(l)[1:(end-dim)]))
end

function preprocess(file_name::String; batch_size=10, dim=70, fragment_size=10000)
    f = jldopen(joinpath(SqState.training_data_path(), file_name), "r")
    points = f["points"]
    𝛒s = f["𝛒s"]

    # 4096 points 1 channel, 10000 data in a data fragment
    xs = reshape(Float32.(points), (4096, 1, fragment_size))

    # 70x70 𝛒, 10000 data in data fragment
    ys = reshape(hcat([𝛒2y(𝛒s[i]) for i in 1:fragment_size]...), (dim*dim, fragment_size))

    return DataLoader((xs, ys), batchsize=batch_size, shuffle=true)
end
