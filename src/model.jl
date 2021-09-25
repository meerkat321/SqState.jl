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

struct gram2ρ end

Flux.@functor gram2ρ

to_complex(𝐱::AbstractArray) = 𝐱[:, :, :, 1] + im.*𝐱[:, :, :, 2]

function ChainRulesCore.rrule(::typeof(to_complex), 𝐱::AbstractArray)
    function to_complex_pullback(𝐲̄)
        return NoTangent(), cat(real.(𝐲̄), imag.(𝐲̄), dims=4)
    end

    return to_complex(𝐱), to_complex_pullback
end

function (m::gram2ρ)(x)
    x = to_complex(Zygote.hook(real, x))
    𝛒 = reshape(Flux.batched_mul(Flux.batched_adjoint(x), x), size(x, 2)^2, 1, :)

    return hcat(real.(𝛒), imag.(𝛒))
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
        Conv((1, ), 64=>6, σ),

        flatten,
        Dense(6*4096, 5*4096, σ),
        Dense(5*4096, 2*100*100),
        x -> reshape(x, 100, 100, :, 2), # gram matrix
        gram2ρ(), # 𝛒

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
