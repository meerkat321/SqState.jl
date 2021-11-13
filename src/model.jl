export
    model_q2args_sqth,
    q2args_sqth_res

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

function q2args_sqth_res()
    σ = relu

    return Chain(
        # stage 0
        # BatchNorm(1),
        Conv((31, ), 1=>8, pad=15),
        BatchNorm(8, σ),

        # res
        res_block((8, 4, 4, 16), (1, 15, 7), (0, 7, 3), 1, 0, 0, σ),
        res_block((16, 8, 8, 32), (1, 15, 7), (0, 7, 3), 1, 0, 0, σ),
        res_block((32, 16, 16, 64), (1, 15, 7), (0, 7, 3), 1, 0, 0, σ),
        # res_block((64, 16, 16, 64), (1, 15, 7), (0, 7, 3), 1, 0, 2, σ),
        # res_block((64, 16, 16, 64), (1, 15, 7), (0, 7, 3), 1, 0, 2, σ),
        res_block((64, 16, 16, 32), (1, 15, 7), (0, 7, 3), 1, 0, 4, σ),
        res_block((32, 8, 8, 16), (1, 15, 7), (0, 7, 3), 1, 0, 4, σ),
        res_block((16, 4, 4, 8), (1, 15, 7), (0, 7, 3), 1, 0, 4, σ),

        # stage 1
        flatten,
        Dense(8*64, 64, σ),
        Dense(64, 16, σ),
        Dense(16, 3, relu)
    )
end
