# using LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(16)
using Dates
using SqState

@time for i in 1:101
    @show i
    @time begin
        file_name="sq_sqth_th_$(replace(string(now()), ':'=>'_'))"
        points, 𝛒s, args = gen_data(n_data=10000)

        data_path = mkpath(SqState.training_data_path())
        jldsave(joinpath(data_path, "$file_name.jld2"); points, 𝛒s, args)

        file = matopen("~/training_data_mat/$file_name.mat", "w")
        write(file, "points", points); write(file, "dms", 𝛒s); write(file, "args", args)
        close(file)
    end
end
