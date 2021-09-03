@testset "preprocess for training" begin
    n = 6
    gen_data(n_data=n, file_name="ci")
    loader = preprocess("ci.jld2", batch_size=2, fragment_size=n)
    x, y = first(loader)

    @test size(x) == (4096, 1, 2)
    @test size(y) == (6, 2)

    rm(joinpath(SqState.training_data_path(), "ci.jld2"))
end

@testset "inference" begin
    # file_name = readdir(joinpath(SqState.data_path(), "Flow"))[end]
    # ρ, w_reshaped = infer(file_name, n_sample=10, fix_θ=true, dim=70)
end
