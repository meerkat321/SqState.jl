using CUDA, Flux, SqState

if has_cuda()
    @info "CUDA is on"
    device = gpu
    CUDA.allowscalar(false)
else
    device = cpu
end

# m = model_ae() |> device

m = Chain(
    SqState.gram2ρ(100)
) |> device

# m(device(rand(Float32, 4096, 1, 5)))

loss(x, y) = Flux.mse(m(x), y)
opt = Flux.Optimiser(WeightDecay(1e-4), Flux.Momentum(1e-2, 0.9))
# Flux.train!(loss, params(m), [(device(rand(Float32, 4096, 1, 5)), device(rand(Float32, 4096, 5)))], opt)
Flux.train!(loss, params(m), [(device(rand(Float32, 100, 100, 5, 2)), device(rand(Float32, 100*100, 2, 5)))], opt)
