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

# function res_block(
#     conv_ch::NTuple{4, <:Integer},
#     conv_kernel_size::NTuple{3, <:Integer},
#     conv_pad::NTuple{3, <:Any},
#     shortcut_kernel_size::Integer,
#     shortcut_pad::Any,
#     pool_size::Integer,
#     σ=leakyrelu
# )
#     conv_layers = Chain(
#         Conv((conv_kernel_size[1], ), conv_ch[1]=>conv_ch[2], pad=conv_pad[1]),
#         BatchNorm(conv_ch[2], σ),
#         Conv((conv_kernel_size[2], ), conv_ch[2]=>conv_ch[3], pad=conv_pad[2]),
#         BatchNorm(conv_ch[3], σ),
#         Conv((conv_kernel_size[3], ), conv_ch[3]=>conv_ch[4], pad=conv_pad[3]),
#         BatchNorm(conv_ch[4]),
#     )

#     shortcut = Chain(
#         Conv((shortcut_kernel_size, ), conv_ch[1]=>conv_ch[end], pad=shortcut_pad),
#         BatchNorm(conv_ch[end])
#     )

#     pool = (pool_size > 0) ? MeanPool((pool_size, )) : identity

#     return Chain(
#         Parallel(+, conv_layers, shortcut),
#         x -> σ.(x),
#         pool,
#         BatchNorm(conv_ch[end], σ)
#     )
# end

function q2args_sqth_res()
    σ = leakyrelu

    return Chain(
        # stage 0
        BatchNorm(1),
        Conv((31, ), 1=>8, pad=15),
        BatchNorm(8, σ),

        # res
        res_block((8, 4, 4, 16), (1, 15, 7), (0, 7, 3), 1, 0, -1, σ),
        res_block((16, 8, 8, 32), (1, 15, 7), (0, 7, 3), 1, 0, -1, σ),
        res_block((32, 16, 16, 64), (1, 15, 7), (0, 7, 3), 1, 0, -1, σ),
        res_block((64, 32, 32, 128), (1, 15, 7), (0, 7, 3), 1, 0, 2, σ),
        res_block((128, 32, 32, 64), (1, 15, 7), (0, 7, 3), 1, 0, 2, σ),
        res_block((64, 16, 16, 32), (1, 15, 7), (0, 7, 3), 1, 0, 2, σ),
        res_block((32, 8, 8, 16), (1, 15, 7), (0, 7, 3), 1, 0, 2, σ),
        res_block((16, 4, 4, 8), (1, 15, 7), (0, 7, 3), 1, 0, 2, σ),

        # stage 1
        flatten,
        Dense(8*128, 64, σ),
        Dense(64, 16, σ),
        Dense(16, 3, relu)
    )
end
