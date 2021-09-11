@testset "training" begin
    n = 6
    for i in 1:2
        points, 𝛒s, args, σs = gen_data(n_data=n)
        jldsave(joinpath(SqState.training_data_path(), "test$i.jld2"); points, 𝛒s, args, σs)
    end

    train("test")

    for i in 1:2
        rm(joinpath(SqState.training_data_path(), "test$i.jld2"))
    end
    rm(joinpath(SqState.model_path(), "test.jld2"))
end
