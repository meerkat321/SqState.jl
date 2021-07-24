using QuantumStateBase
using SqState

# m = get_model("model0")

new_state = SqueezedThermalState(ξ(1.34, 3.7), 0.08)
# Base.print_matrix(IOContext(stdout, :limit=>true), 𝛒(new_state))

new_data = reshape(Float32.(rand(new_state, 4096, IsGaussian)[2, :]), (4096, 1, 1))

l_new = reshape(m(new_data), 4900)
𝛒_new = post_processing(l_new)
# Base.print_matrix(IOContext(stdout, :limit=>true), 𝛒_new)

sum(𝛒(new_state) - 𝛒_new)

# @show size(s_model()(new_data))
