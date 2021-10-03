export
    model_q2ρ_sqth_th,
    model_q2args_sqth_th,
    model_q2args_sqth

function model_q2ρ_sqth_th()
    modes = (12, )
    ch = 32=>32
    σ = leakyrelu
    dim = 100

    return Chain(
        Dense(2, ch[1]),
        FourierOperator(ch, modes, σ),
        FourierOperator(ch, modes, σ),
        FourierOperator(ch, modes, σ),
        FourierOperator(ch, modes),
        Dense(ch[2], 128, σ),
        Dense(128, 1),

        flatten,
        Dense(4096, dim*dim), # cholesky
        cholesky2ρ,
    )
end

function model_q2args_sqth_th()
    modes = (12, )
    ch = 32=>32
    σ = leakyrelu

    return Chain(
        Conv((1, ), 1=>32),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, permuted=true),
        Conv((1, ), 32=>128, σ),
        Conv((1, ), 128=>1),

        flatten,
        Dense(4096, 1024, gelu),
        Dense(1024, 256, gelu),
        Dense(256, 64, gelu),
        Dense(64, 6, relu),
    )
end

function model_q2args_sqth()
    modes = (12, )
    ch = 64
    σ = leakyrelu

    return Chain(
        Dense(2, ch),
        FourierOperator(ch=>ch, modes, σ),
        FourierOperator(ch=>ch, modes, σ),
        FourierOperator(ch=>ch, modes, σ),
        FourierOperator(ch=>ch, modes),
        Dense(ch, 128, σ),
        Dense(128, 1),

        flatten,
        Dense(4096, 3),
    )
end
