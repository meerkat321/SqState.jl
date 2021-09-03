export model

function DenseCh(ch::Pair{S, S}, σ=identity, p::S=0) where {S <: Integer}
    pool = (p > 0) ? MaxPool((p, )) : identity

    return Chain(
        Conv((1, ), ch, σ),
        pool,
        BatchNorm(ch[end]),
    )
end

function model()
    modes = (64, )
    σ = gelu

    return Chain(
        # stage 0
        Conv((1, ), 1=>64, σ),

        # fourier operator
        FourierOperator(64=>64, modes, σ, permuted=true),
        MeanPool((4, )),
        FourierOperator(64=>64, modes, σ, permuted=true),
        MeanPool((4, )),
        FourierOperator(64=>64, modes, σ, permuted=true),
        MeanPool((4, )),
        FourierOperator(64=>64, modes, permuted=true),

        # stage 1
        Conv((1, ), 64=>8, σ),
        flatten,
        Dense(8*64, 64, σ),
        Dense(64, 16, σ),
        Dense(16, 6, relu)
    )
end
