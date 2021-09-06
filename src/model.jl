export model

function model()
    modes = (24, )
    ch = 64=>64
    σ = gelu

    return Chain(
        Conv((1, ), 1=>64),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, σ, permuted=true),
        FourierOperator(ch, modes, permuted=true),

        Conv((2, ), 64=>32, σ, stride=2),
        Conv((2, ), 32=>16, σ, stride=2),
        Conv((4, ), 16=>8, σ, stride=4),
        Conv((4, ), 8=>4, σ, stride=4),

        flatten,
        Dense(4*64, 32, σ),
        Dense(32, 6, relu),
    )
end

function old_model()
    function conv_layers(ch::NTuple{4, <:Integer}, kernel_size::NTuple{3, <:Integer}, pad::NTuple{3, <:Any})
        return Chain(
            Conv((kernel_size[1], ), ch[1]=>ch[2], pad=pad[1]),
            BatchNorm(ch[2], relu),
            Conv((kernel_size[2], ), ch[2]=>ch[3], pad=pad[2]),
            BatchNorm(ch[3], relu),
            Conv((kernel_size[3], ), ch[3]=>ch[4], pad=pad[3]),
            BatchNorm(ch[4]),
        )
    end

    function shortcut(ch::NTuple{2, <:Integer}, kernel_size::Integer, pad::Any)
        return Chain(
            Conv((kernel_size, ), ch[1]=>ch[2], pad=pad),
            BatchNorm(ch[2])
        )
    end

    function res_block(
        conv_ch::NTuple{4, <:Integer},
        conv_kernel_size::NTuple{3, <:Integer},
        conv_pad::NTuple{3, <:Any},
        shortcut_kernel_size::Integer,
        shortcut_pad::Any,
        pool_size::Integer;
    )
        pool = (pool_size > 0) ? MeanPool((pool_size, )) : identity

        return Chain(
            Parallel(+,
                conv_layers(conv_ch, conv_kernel_size, conv_pad),
                shortcut((conv_ch[1], conv_ch[end]), shortcut_kernel_size, shortcut_pad),
            ),
            x -> relu.(x),
            pool,
            BatchNorm(conv_ch[end], relu)
        )
    end

    return Chain(
        # stage 0
        BatchNorm(1),
        Conv((31, ), 1=>8, pad=15),
        BatchNorm(8, relu),

        # res
        res_block((8, 4, 4, 16), (1, 15, 7), (0, 7, 3), 1, 0, -1),
        res_block((16, 8, 8, 32), (1, 15, 7), (0, 7, 3), 1, 0, -1),
        res_block((32, 16, 16, 64), (1, 15, 7), (0, 7, 3), 1, 0, -1),
        res_block((64, 32, 32, 128), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((128, 32, 32, 64), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((64, 16, 16, 32), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((32, 8, 8, 16), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((16, 4, 4, 8), (1, 15, 7), (0, 7, 3), 1, 0, 2),

        # stage 1
        flatten,
        Dense(8*128, 64, relu),
        Dense(64, 16, relu),
        Dense(16, 6, relu)
    )
end

# function model()
#     function FourierBlock(ch, modes, σ)
#         return Chain(
#             FourierOperator(ch=>ch, modes, σ, permuted=true),
#             FourierOperator(ch=>ch, modes, σ, permuted=true),
#             FourierOperator(ch=>ch, modes, σ, permuted=true),
#             FourierOperator(ch=>ch, modes, σ, permuted=true),
#             FourierOperator(ch=>ch, modes, σ, permuted=true),
#             FourierOperator(ch=>ch, modes, permuted=true),
#         )
#     end

#     function res_block(
#         ch::NTuple{4, T},
#         conv_kernel_size::NTuple{3, T},
#         conv_pad::NTuple{3, T},
#         shortcut_kernel_size::T,
#         shortcut_pad::T,
#         stride::T,
#         σ=identity
#     ) where {T<:Integer}
#         conv_layers = Chain(
#             Conv((conv_kernel_size[1], ), ch[1]=>ch[2], pad=conv_pad[1], stride=stride),
#             BatchNorm(ch[2], σ),
#             Conv((conv_kernel_size[2], ), ch[2]=>ch[3], pad=conv_pad[2]),
#             BatchNorm(ch[3], σ),
#             Conv((conv_kernel_size[3], ), ch[3]=>ch[4], pad=conv_pad[3]),
#             BatchNorm(ch[4]),
#         )

#         shortcut = Chain(
#             Conv((shortcut_kernel_size, ), ch[1]=>ch[end], pad=shortcut_pad, stride=stride),
#             BatchNorm(ch[end])
#         )

#         return Chain(
#             Parallel(+, conv_layers, shortcut),
#             x -> σ.(x),
#             BatchNorm(ch[end], σ)
#         )
#     end

#     modes = (24, )
#     ch = 64=>64
#     σ = gelu

#     return Chain(
#         Conv((1, ), 1=>64),
#         FourierOperator(ch, modes, σ, permuted=true),
#         FourierOperator(ch, modes, σ, permuted=true),
#         FourierOperator(ch, modes, σ, permuted=true),
#         FourierOperator(ch, modes, permuted=true),

#         Parallel(+,
#             Chain(
#                 Conv((1, ), 64=>16, σ),
#                 Conv((5, ), 16=>16, σ, stride=4),
#                 Conv((1, ), 16=>32, σ),
#             ),
#             Conv((5, ), 64=>32, σ, stride=4)
#         ),
#         Parallel(+,
#             Chain(
#                 Conv((1, ), 32=>8, σ),
#                 Conv((5, ), 8=>8, σ, stride=4),
#                 Conv((1, ), 8=>16, σ),
#             ),
#             Conv((5, ), 32=>16, σ, stride=4)
#         ),
#         Parallel(+,
#             Chain(
#                 Conv((1, ), 16=>4, σ),
#                 Conv((5, ), 4=>4, σ, stride=4),
#                 Conv((1, ), 4=>8, σ),
#             ),
#             Conv((5, ), 16=>8, σ, stride=4)
#         ),
#         Parallel(+,
#             Chain(
#                 Conv((1, ), 8=>2, σ),
#                 Conv((5, ), 2=>2, σ, stride=4),
#                 Conv((1, ), 2=>4, σ),
#             ),
#             Conv((5, ), 8=>4, σ, stride=4)
#         ),

#         flatten,
#         Dense(4*(16-1), 32, σ),
#         Dense(32, 6, relu)
#     )
# end
