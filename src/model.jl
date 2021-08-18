export model

function fourier_block(ch::Pair{S, S}, modes::NTuple{N, S}, pool_size::S, σ=gelu) where {N, S<:Integer}
    pool = (pool_size > 0) ? MaxPool((pool_size, )) : identity

    return Chain(
        FourierOperator(ch, modes, σ, permuted=true),
        pool,
        BatchNorm(ch[end], σ),
    )
end

function model()
    modes = (64, )

    return Chain(
        # stage 0
        BatchNorm(1),

        # fourier operator
        fourier_block(1=>8, modes, -1),
        fourier_block(8=>16, modes, -1),
        fourier_block(16=>32, modes, -1),
        fourier_block(32=>64, modes, 2),
        fourier_block(64=>128, modes, 2),
        fourier_block(128=>64, modes, 2),
        fourier_block(64=>32, modes, 2),
        fourier_block(32=>16, modes, 2),
        fourier_block(16=>8, modes, 2),

        # stage 1
        flatten,
        Dense(8*64, 64, σ),
        Dense(64, 16, σ),
        Dense(16, 8, σ),
        Dense(8, 3, relu)
    )
end
