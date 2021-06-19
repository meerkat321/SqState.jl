using SqState
using DataDeps
using JLD2

function start(;nr, nθ, nn̄, n=40960, dim=500, file_name="training_data")
    data = Dict([
        [r, θ, n̄]=>Matrix{Float64}(undef, n, 2)
        for r in LinRange(0, 16, nr), θ in LinRange(0, π/2, nθ), n̄ in LinRange(0, 0.5, nn̄)
    ])

    for ((r, θ, n̄), points) in data
        gen_gaussian_training_data!(points, SqueezedThermalState(ξ(r, θ), n̄, dim=dim))
    end

    data_path = mkpath(joinpath(datadep"SqState", "training_data"))
    jldsave(joinpath(data_path, "$file_name.jld2"); data)
end

# jit
start(;nr=2, nθ=2, nn̄=2, file_name="jit")

# generate training data
@time start(;nr=35, nθ=60, nn̄=50)
