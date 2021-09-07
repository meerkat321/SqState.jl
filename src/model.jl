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
