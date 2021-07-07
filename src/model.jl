using SqState
using LinearAlgebra
using Flux
using CUDA
using JLD2

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(true)
end

dim = 70

function loss(ŷ, 𝐲)
    l = sum(
        diagm((i-dim) => ŷ[(sum(1:(i-1))+1):sum(1:i)])
        for i in 1:dim
    )

    return Flux.mse(l * l', 𝐲)
end

file_names = readdir(SqState.training_data_path())
f = jldopen(joinpath(SqState.training_data_path(), file_names[1]), "r")
points = f["points"]
𝛒s = f["𝛒s"]

for i in 1:1
    x = Float32.(points[:, i])
    y = ComplexF32.(𝛒s[i])

    y_dummy = rand(ComplexF32, sum(1:70))
    println(loss(y_dummy, y))
end
