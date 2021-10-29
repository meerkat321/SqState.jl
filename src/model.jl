export
    model_q2args_sqth,
    cnn_q2args_sqth

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
        Dense(4096, 3, relu),
    )
end

function block(kernel_size::NTuple{4}, ch::NTuple{5}, stride::NTuple{3}, pad::NTuple{3})
    return Chain(
        Parallel(hcat,
            Chain(
                BatchNorm(ch[1], σ),
                Conv((kernel_size[1], ), ch[1]=>ch[2], stride=stride[1], pad=pad[1]),
                BatchNorm(ch[2], σ),
                Conv((kernel_size[2], ), ch[2]=>ch[3], stride=stride[2], pad=pad[2]),
            ),
            Chain(
                x -> σ.(x),
                Conv((kernel_size[3], ), ch[1]=>ch[4], stride=stride[3], pad=pad[3])
            )
        ),
        BatchNorm(ch[3]+ch[4], σ),
        Conv((kernel_size[4], ), ch[3]+ch[4]=>ch[5]),
    )
end

function cnn_q2args_sqth()
    σ = leakyrelu

    return Chain(
        BatchNorm(1, σ),
        Parallel(hcat,
            Chain(
                Conv((31, ), 1=>64, pad=15),
                BatchNorm(64, σ),
                Conv((31, ), 64=>64, pad=15),
            ),
            Conv((31, ), 1=>64, pad=15)
        ),
        BatchNorm(128, σ),
        Conv((1, ), 128=>48),

        block((16, 15, 1, 1), (48, 24, 24, 48, 48), (2, 1, 2), (7, 7, 0)),
        block((4, 3, 1, 1), (48, 32, 32, 48, 64), (4, 1, 4), (1, 1, 0)),
        block((4, 3, 1, 1), (64, 72, 72, 64, 32), (4, 1, 4), (1, 1, 0)),
        block((4, 3, 1, 4), (32, 16, 8, 32, 5), (4, 1, 4), (1, 1, 0)),

        flatten,
        Dense(145, 6, σ)
    )
end
