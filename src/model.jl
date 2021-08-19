export model

function fourier_block(ch::Pair{S, S}, modes::NTuple{N, S}, pool_size::S, σ) where {N, S<:Integer}
    pool = (pool_size > 0) ? MeanPool((pool_size, )) : identity

    return Chain(
        FourierOperator(ch, modes, σ, permuted=true),
        pool,
        BatchNorm(ch[end], σ),
    )
end

function model()
    modes = (64, )
    σ = gelu

    return Chain(
        # stage 0
        BatchNorm(1),
        Conv((3, ), 1=>64, pad=1, σ),

        # fourier operator
        fourier_block(64=>64, modes, 0, σ),
        fourier_block(64=>64, modes, 2, σ),
        fourier_block(64=>32, modes, 2, σ),
        fourier_block(32=>16, modes, 4, σ),
        fourier_block(16=>8, modes, 4, σ),

        # stage 1
        flatten,
        Dense(8*64, 64, σ),
        Dense(64, 16, σ),
        Dense(16, 3, relu)
    )
end
