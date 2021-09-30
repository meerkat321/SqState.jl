export
    model_q2ρ_sqth_th,
    model_q2args_sqth_th,
    model_q2args_sqth

function model_q2ρ_sqth_th()
    modes = (12, )
    ch = 64=>64
    σ = leakyrelu
    dim = 70

    return Chain(
        Conv((1, ), 1=>64),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, permuted=true),
        Conv((1, ), 64=>128, σ),
        Conv((1, ), 128=>1),

        flatten,
        Dense(4096, dim*dim), # cholesky
        cholesky2ρ,
    )
end

function model_q2args_sqth_th()
    modes = (12, )
    ch = 64=>64
    σ = leakyrelu

    return Chain(
        Conv((1, ), 1=>64),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, permuted=true),
        Conv((1, ), 64=>128, σ),
        Conv((1, ), 128=>1),

        flatten,
        Dense(4096, 6, relu),
    )
end

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
