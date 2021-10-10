export model_q2args_sqth

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
        Dense(ch, 4ch, σ),
        Dense(4ch, 1),

        flatten,
        Dense(4096, 3),
    )
end
