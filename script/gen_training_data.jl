using LinearAlgebra
using SqState

# LinearAlgebra.BLAS.set_num_threads(16)

# jit
@time gen_squeezed_thermal_data(n_data=100, file_name=nothing)

# generate training data
# about 680 sec for 10000 data
# about 18.9 hr for 100 batch files
@time for i in 1:100
    @show i
    @time gen_squeezed_thermal_data(n_data=10000)
end
