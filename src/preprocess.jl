function preprocess_q2args(file_name::String; batch_size=50)
    f = jldopen(joinpath(SqState.training_data_path(), file_name), "r")
    points = f["points"][2, :, :]

    # 4096 points 1 channel, 10000 data in a data fragment
    xs = reshape(Float32.(points), (4096, 1, :))

    # std, 10000 data in data fragment
    ys = f["σs"]

    close(f)

    return DataLoader((xs, ys), batchsize=batch_size, shuffle=true)
end
