# using LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(16)
using Dates
using SqState
using JLD2
using MAT

function gen_data(prefix::String, gen_func)
    @time for i in 1:101
        @show i
        @time begin
            file_name="$(prefix)_$(replace(string(now()), ':'=>'_'))"
            points, 𝛒s, args, σs = gen_func(n_data=10000)

            data_path = mkpath(joinpath(SqState.training_data_path(), prefix))
            jldsave(joinpath(data_path, "$file_name.jld2"); points, 𝛒s, args, σs)

            data_path = mkpath("training_data_mat/$prefix")
            file = matopen(joinpath(data_path, "$file_name.mat"), "w")
            write(file, "points", points); write(file, "dms", 𝛒s); write(file, "args", args); write(file, "stds", σs)
            close(file)
        end
    end
end

gen_data("sqth_th", gen_data_sqth_th)

gen_data("sqth", gen_data_sqth)
