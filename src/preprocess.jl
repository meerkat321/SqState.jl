function preprocess_q2σs(file::String; batch_size=50)
    f = jldopen(file, "r")
    points = f["points"][2, :, :]

    # 4096 points 1 channel, 10000 data in a data fragment
    xs = reshape(Float32.(points), (4096, 1, :))

    # σs, 10000 data in data fragment
    ys = f["σs"]

    close(f)

    return DataLoader((xs, ys), batchsize=batch_size, shuffle=true)
end


function preprocess_q2ρ(file::String; batch_size=50, dim=100)
    f = jldopen(file, "r")
    points = f["points"]

    # 2*4096 points, 10000 data in a data fragment
    xs = Float32.(points)

    # 𝛒s, 100x100, 10000 data in data fragment
    ys = reshape(f["𝛒s"][1:dim, 1:dim, :], dim*dim, 1, :)
    ys = hcat(real.(ys), imag(ys))

    close(f)

    return DataLoader((xs, ys), batchsize=batch_size, shuffle=true)
end

function preprocess_q2args(file::String; batch_size=50)
    f = jldopen(file, "r")
    points = f["points"]

    # 4096 points 1 channel, 10000 data in a data fragment
    xs = Float32.(points)

    # r, θ, n̄, 10000 data in data fragment
    ys = f["args"]

    close(f)

    return DataLoader((xs, ys), batchsize=batch_size, shuffle=true)
end
