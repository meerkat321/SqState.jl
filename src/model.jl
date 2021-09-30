using Zygote, LinearAlgebra, ChainRulesCore

export
    model_q2args_sqth_th,
    model_q2args_sqth

function model_q2args_sqth()
    modes = (12, )
    ch = 32=>32
    σ = gelu

    return Chain(
        Conv((1, ), 1=>32),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, permuted=true),
        Conv((1, ), 32=>128, σ),
        Conv((1, ), 128=>1),

        flatten,
        Dense(4096, 1024, σ),
        Dense(1024, 256, σ),
        Dense(256, 64, σ),
        Dense(64, 3, relu),
    )
end

function model_q2args_sqth_th()
    modes = (12, )
    ch = 32=>32
    σ = gelu

    return Chain(
        Conv((1, ), 1=>32),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, permuted=true),
        Conv((1, ), 32=>128, σ),
        Conv((1, ), 128=>1),

        flatten,
        Dense(4096, 1024, σ),
        Dense(1024, 256, σ),
        Dense(256, 64, σ),
        Dense(64, 6, relu),
    )
end
