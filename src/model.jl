using Zygote, LinearAlgebra

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

struct gram2ρ
    dim::Int64
end

Flux.@functor gram2ρ

function (m::gram2ρ)(x)
    x = Zygote.hook(real, x)

    x = x[:, :, :, 1] + im.*x[:, :, :, 2]
    𝛒 = reshape(Flux.batched_mul(x, Flux.batched_transpose(x)), m.dim*m.dim, 1, :)

    return cat(real.(𝛒), imag.(𝛒), dims=2)
end

function model_ae()
    modes = (24, )
    ch = 64=>64
    σ = gelu

    return Chain(
        Conv((1, ), 1=>64),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, permuted=true),
        Conv((1, ), 64=>6, σ,),

        flatten,
        Dense(6*4096, 5*4096, σ),
        Dense(5*4096, 2*100*100),
        x -> reshape(x, 100, 100, :, 2), # gram matrix
        gram2ρ(100), # 𝛒

        # enbading (dim*dim, 2, batch)

        Conv((1, ), 2=>64),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, permuted=true),
        Conv((1, ), 64=>1, σ,),

        flatten,
        Dense(100*100, 2*4096, σ),
        Dense(2*4096, 4096, relu), # std
    )
end
