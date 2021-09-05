export model

function FourierBlock(ch, modes, σ)
    return Chain(
        FourierOperator(ch=>ch, modes, σ, permuted=true),
        FourierOperator(ch=>ch, modes, σ, permuted=true),
        FourierOperator(ch=>ch, modes, σ, permuted=true),
        FourierOperator(ch=>ch, modes, σ, permuted=true),
        FourierOperator(ch=>ch, modes, σ, permuted=true),
        FourierOperator(ch=>ch, modes, permuted=true),
    )
end

function res_block(
    ch::NTuple{4, T},
    conv_kernel_size::NTuple{3, T},
    conv_pad::NTuple{3, T},
    shortcut_kernel_size::T,
    shortcut_pad::T,
    stride::T,
    σ=identity
) where {T<:Integer}
    conv_layers = Chain(
        Conv((conv_kernel_size[1], ), ch[1]=>ch[2], pad=conv_pad[1], stride=stride),
        BatchNorm(ch[2], σ),
        Conv((conv_kernel_size[2], ), ch[2]=>ch[3], pad=conv_pad[2]),
        BatchNorm(ch[3], σ),
        Conv((conv_kernel_size[3], ), ch[3]=>ch[4], pad=conv_pad[3]),
        BatchNorm(ch[4]),
    )

    shortcut = Chain(
        Conv((shortcut_kernel_size, ), ch[1]=>ch[end], pad=shortcut_pad, stride=stride),
        BatchNorm(ch[end])
    )

    return Chain(
        Parallel(+, conv_layers, shortcut),
        x -> σ.(x),
        BatchNorm(ch[end], σ)
    )
end

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
        Dense(32, 6, relu)
    )
end

# function model()
#     modes = (64, )
#     σ = gelu

#     return Chain(
#         # stage 0
#         Conv((1, ), 1=>64, σ),

#         FourierBlock(64, modes, σ),
#         res_block((64, 16, 16, 32), (4, 15, 7), (0, 7, 3), 4, 0, 4, σ),
#         res_block((32, 8, 8, 16), (4, 15, 7), (0, 7, 3), 4, 0, 4, σ),
#         res_block((16, 4, 4, 8), (4, 15, 7), (0, 7, 3), 4, 0, 4, σ),

#         # stage 1
#         flatten,
#         Dense(8*64, 64, σ),
#         Dense(64, 16, σ),
#         Dense(16, 6, relu)
#     )
# end
