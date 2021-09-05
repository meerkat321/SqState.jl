export model

function model()
    modes = (64, )
    σ = gelu

    return Chain(
        Conv((1, ), 1=>64),
        FourierOperator(64=>64, modes, σ, permuted=true),
        FourierOperator(64=>64, modes, σ, permuted=true),
        FourierOperator(64=>64, modes, σ, permuted=true),
        FourierOperator(64=>64, modes, permuted=true),
        Conv((1, ), 64=>1),

        flatten,
        Dense(4096, 16384, σ),
        Dense(16384, 100*100),
        x -> x ./ sqrt(max(sum(x.^2), 1f-12)) # l-2 normalize
    )
end
