export model

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

function model()
    return Chain(
        # stage 0
        BatchNorm(1),
        Conv((31, ), 1=>8, pad=15),
        BatchNorm(8, relu),

        # res
        res_block((8, 4, 4, 16), (1, 15, 7), (0, 7, 3), 1, 0, -1),
        res_block((16, 8, 8, 32), (1, 15, 7), (0, 7, 3), 1, 0, -1),
        res_block((32, 16, 16, 64), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((64, 32, 32, 128), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((128, 32, 32, 64), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((64, 16, 16, 32), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((32, 8, 8, 16), (1, 15, 7), (0, 7, 3), 1, 0, 2),
        res_block((16, 4, 4, 8), (1, 15, 7), (0, 7, 3), 1, 0, 2),

        # stage 1
        flatten,
        Dense(8*64, 64, relu),
        Dense(64, 16, relu),
        Dense(16, 3, relu)
    )
end
