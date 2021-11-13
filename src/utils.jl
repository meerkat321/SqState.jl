########
# path #
########

training_data_path() = mkpath(joinpath(datadep"SqState", "training_data"))
model_path() = mkpath(joinpath(datadep"SqState", "model"))

############
# gen data #
############

rand2range(x_range) = x_range[1] + (x_range[2]-x_range[1])*rand()

function rand_arg_sqth_th(r_range, θ_range, n̄_range)
    r = rand2range(r_range)
    θ = rand2range(θ_range)
    n̄ = rand2range(n̄_range)
    n̄0 = rand2range(n̄_range)
    c1 = rand()
    c2 = 1 - c1

    return r, θ, n̄, n̄0, c1, c2
end

function rand_arg_sqth(r_range, θ_range, n̄_range)
    r = rand2range(r_range)
    θ = rand2range(θ_range)
    n̄ = rand2range(n̄_range)

    return r, θ, n̄
end

function construct_state_sqth_th(r, θ, n̄, n̄0, c1, c2, dim)
    state =
        c1 * SqueezedThermalState(ξ(r, θ), n̄, dim=dim) +
        c2 * ThermalState(n̄0, dim=dim)

    return state
end

function construct_state_sqth(r, θ, n̄, dim)
    return SqueezedThermalState(ξ(r, θ), n̄, dim=dim)
end

#########
# model #
#########

to_complex(𝐱1::AbstractArray, 𝐱2::AbstractArray) = 𝐱1 + im.*𝐱2

function ChainRulesCore.rrule(::typeof(to_complex), 𝐱1::AbstractArray, 𝐱2::AbstractArray)
    function to_complex_pullback(𝐲̄)
        return NoTangent(), real.(𝐲̄), imag.(𝐲̄)
    end

    return to_complex(𝐱1, 𝐱2), to_complex_pullback
end

function reshape_cholesky(x)
    dim = Int(sqrt(size(x, 1)))
    𝐱_row = reshape(x, dim, dim, :)
    𝐱_real = cat([reshape(tril(𝐱_row[:, :, i]), dim, dim, 1) for i in axes(𝐱_row, 3)]..., dims=3)
    𝐱_imag = cat([reshape(tril(𝐱_row[:, :, i]', -1), dim, dim, 1) for i in axes(𝐱_row, 3)]..., dims=3)
    𝐱 = to_complex(𝐱_real, 𝐱_imag)

    return 𝐱
end

function cholesky2ρ(x)
    𝐱 = reshape_cholesky(Zygote.hook(real, x))
    𝛒 = Flux.batched_mul(𝐱, Flux.batched_adjoint(𝐱))
    𝛒 = cat([reshape(𝛒[:, :, i]/tr(𝛒[:, :, i]), 1, size(𝛒, 1), size(𝛒, 2), 1) for i in axes(𝛒, 3)]..., dims=4)

    return vcat(real.(𝛒), imag.(𝛒))
end

function res_block(
    ch::NTuple{4, <:Integer},
    conv_kernel_size::NTuple{3, <:Integer},
    conv_pad::NTuple{3, <:Any},
    shortcut_kernel_size::Integer,
    shortcut_pad::Any,
    pool_size::Integer,
    σ=identity
)
    conv_layers = Chain(
        Conv((conv_kernel_size[1], ), ch[1]=>ch[2], σ,  pad=conv_pad[1]),
        # BatchNorm(ch[2], σ),
        Conv((conv_kernel_size[2], ), ch[2]=>ch[3], σ, pad=conv_pad[2]),
        # BatchNorm(ch[3], σ),
        Conv((conv_kernel_size[3], ), ch[3]=>ch[4], σ, pad=conv_pad[3]),
        # BatchNorm(ch[4]),
    )
    shortcut = Chain(
        Conv((shortcut_kernel_size, ), ch[1]=>ch[end], σ, pad=shortcut_pad),
        # BatchNorm(ch[end])
    )
    pool = (pool_size > 0) ? MeanPool((pool_size, )) : identity

    return Chain(
        Parallel(+, conv_layers, shortcut),
        x -> σ.(x),
        pool,
        BatchNorm(ch[end], σ)
    )
end

############
# training #
############

function update_model!(model_path::String, model_name::String, model)
    model = cpu(model)
    jldsave(joinpath(model_path, "$model_name.jld2"); model)
    @warn "'$model_name' updated!"
end

function get_device()
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    return device
end

#############
# inference #
#############

function get_model(model_name::String)
    f = jldopen(joinpath(model_path() , "$model_name.jld2"))
    model = f["model"]
    close(f)

    return model
end
