using Zygote, LinearAlgebra, ChainRulesCore

export
    model,
    model_ae,
    model_q2ρ

# function model()
#     modes = (24, )
#     ch = 64=>64
#     σ = gelu

#     return Chain(
#         Conv((1, ), 1=>64),
#         FourierOperator(ch, modes, σ, permuted=true),
#         FourierOperator(ch, modes, σ, permuted=true),
#         FourierOperator(ch, modes, σ, permuted=true),
#         FourierOperator(ch, modes, permuted=true),

#         Conv((2, ), 64=>32, σ, stride=2),
#         Conv((2, ), 32=>16, σ, stride=2),
#         Conv((4, ), 16=>8, σ, stride=4),
#         Conv((4, ), 8=>4, σ, stride=4),

#         flatten,
#         Dense(4*64, 32, σ),
#         Dense(32, 6, relu),
#     )
# end

to_complex(𝐱1::AbstractArray, 𝐱2::AbstractArray) = 𝐱1 + im.*𝐱2

function ChainRulesCore.rrule(::typeof(to_complex), 𝐱1::AbstractArray, 𝐱2::AbstractArray)
    function to_complex_pullback(𝐲̄)
        return NoTangent(), real.(𝐲̄), imag.(𝐲̄)
    end

    return to_complex(𝐱1, 𝐱2), to_complex_pullback
end

struct Cholesky2ρ end

Flux.@functor Cholesky2ρ

function reshape_cholesky(x)
    dim = Int(sqrt(size(x, 1)))
    𝐱_row = reshape(x, dim, dim, :)
    𝐱_real = cat([reshape(tril(𝐱_row[:, :, i]), dim, dim, 1) for i in axes(𝐱_row, 3)]..., dims=3)
    𝐱_imag = cat([reshape(tril(𝐱_row[:, :, i]', -1), dim, dim, 1) for i in axes(𝐱_row, 3)]..., dims=3)
    𝐱 = to_complex(𝐱_real, 𝐱_imag)

    return 𝐱
end

function (m::Cholesky2ρ)(x)
    𝐱 = reshape_cholesky(Zygote.hook(real, x))
    𝛒 = Flux.batched_mul(𝐱, Flux.batched_adjoint(𝐱))
    𝛒 = cat([reshape(𝛒[:, :, i]/tr(𝛒[:, :, i]), size(𝛒, 1), size(𝛒, 2), 1) for i in axes(𝛒, 3)]..., dims=3)
    𝛒 = reshape(𝛒, size(𝛒, 1)*size(𝛒, 2), 1, :)

    return hcat(real.(𝛒), imag.(𝛒))
end

# function (m::Cholesky2ρ)(x)
#     𝐱 = reshape_cholesky(Zygote.hook(real, x))
#     𝛒 = reshape(Flux.batched_mul(𝐱, Flux.batched_adjoint(𝐱)), size(𝐱, 1)^2, 1, :)

#     return hcat(real.(𝛒), imag.(𝛒))
# end

l2_norm(x) = x ./ sqrt(max(sum(x.^2), 1f-12))

# function model_ae()
#     modes = (24, )
#     ch = 64=>64
#     σ = gelu
#     dim = 70

#     return Chain(
#         Conv((1, ), 1=>64),
#         FourierOperator(ch, modes, σ, permuted=true),
#         FourierOperator(ch, modes, σ, permuted=true),
#         FourierOperator(ch, modes, σ, permuted=true),
#         FourierOperator(ch, modes, permuted=true),
#         Conv((1, ), 64=>4),

#         flatten,
#         Dense(4*4096, 2*4096, σ),
#         Dense(2*4096, dim*dim), # cholesky
#         # l2_norm, # l-2 normalize
#         Cholesky2ρ(),

#         # enbading (dim*dim, 2, batch)

#         Conv((1, ), 2=>64),
#         FourierOperator(ch, modes, σ, permuted=true),
#         FourierOperator(ch, modes, σ, permuted=true),
#         FourierOperator(ch, modes, σ, permuted=true),
#         FourierOperator(ch, modes, permuted=true),
#         Conv((1, ), 64=>4),

#         flatten,
#         Dense(4*dim*dim, 2*4096, σ),
#         Dense(2*4096, 4096), # std
#     )
# end

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
        Conv((conv_kernel_size[1], ), ch[1]=>ch[2], pad=conv_pad[1]),
        BatchNorm(ch[2], σ),
        Conv((conv_kernel_size[2], ), ch[2]=>ch[3], pad=conv_pad[2]),
        BatchNorm(ch[3], σ),
        Conv((conv_kernel_size[3], ), ch[3]=>ch[4], pad=conv_pad[3]),
        BatchNorm(ch[4]),
    )
    shortcut = Chain(
        Conv((shortcut_kernel_size, ), ch[1]=>ch[end], pad=shortcut_pad),
        BatchNorm(ch[end])
    )
    pool = (pool_size > 0) ? MaxPool((pool_size, )) : identity

    return Chain(
        Parallel(+, conv_layers, shortcut),
        x -> σ.(x),
        pool,
        BatchNorm(ch[end], σ)
    )
end

function model_q2ρ()
    modes = (12, )
    ch = 64=>64
    σ = gelu
    dim = 100

    return Chain(
        Conv((1, ), 1=>64),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, permuted=true),
        Conv((1, ), 64=>128, σ),
        Conv((1, ), 128=>4),

        # BatchNorm(8, σ),
        # res_block((8, 4, 4, 16), (1, 15, 7), (0, 7, 3), 1, 0, 2, σ),
        # res_block((16, 8, 8, 32), (1, 15, 7), (0, 7, 3), 1, 0, 2, σ),
        # res_block((32, 8, 8, 16), (1, 15, 7), (0, 7, 3), 1, 0, 2, σ),
        # res_block((16, 4, 4, 8), (1, 15, 7), (0, 7, 3), 1, 0, 2, σ),

        flatten,
        Dense(4*4096, 3*4096, σ),
        Dense(3*4096, 3*4096, σ),
        Dense(3*4096, 3*4096, σ),
        Dense(3*4096, dim*dim), # cholesky
        Cholesky2ρ(),
    )
end
