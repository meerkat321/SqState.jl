using Zygote, LinearAlgebra, ChainRulesCore

export
    model,
    model_ae

function model()
    modes = (24, )
    ch = 64=>64
    σ = gelu

    return Chain(
        Conv((1, ), 1=>64),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, permuted=true),

        Conv((2, ), 64=>32, σ, stride=2),
        Conv((2, ), 32=>16, σ, stride=2),
        Conv((4, ), 16=>8, σ, stride=4),
        Conv((4, ), 8=>4, σ, stride=4),

        flatten,
        Dense(4*64, 32, σ),
        Dense(32, 6, relu),
    )
end

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
    𝛒 = reshape(Flux.batched_mul(𝐱, Flux.batched_adjoint(𝐱)), size(𝐱, 1)^2, 1, :)

    return hcat(real.(𝛒), imag.(𝛒))
end

l2_norm(x) = x ./ sqrt(max(sum(x.^2), 1f-12))

function model_ae()
    modes = (24, )
    ch = 64=>64
    σ = gelu
    dim = 70

    return Chain(
        Conv((1, ), 1=>64),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, permuted=true),
        Conv((1, ), 64=>4),

        flatten,
        Dense(4*4096, 2*4096, σ),
        Dense(2*4096, dim*dim), # cholesky
        # l2_norm, # l-2 normalize
        Cholesky2ρ(),

        # enbading (dim*dim, 2, batch)

        Conv((1, ), 2=>64),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, permuted=true),
        Conv((1, ), 64=>4),

        flatten,
        Dense(4*dim*dim, 2*4096, σ),
        Dense(2*4096, 4096), # std
    )
end
