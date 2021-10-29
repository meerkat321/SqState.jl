using SqState

m = cnn_q2args_sqth()

println(
    size(m(rand(Float32, 4096, 1, 5)))
)
