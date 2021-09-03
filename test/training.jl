@testset "training" begin
    # n = 6
    # gen_squeezed_thermal_data(
    #     n_data=6, n_points=4096,
    #     r_range=(0, 2), θ_range=(0, 2π), n̄_range=(0, 0.5), θ_offset_range=(0, 0),
    #     point_dim=500, label_dim=70,
    #     file_name="ci"
    # )

    # m, in_losses, out_losses = training_process(
    #     "ci",
    #     data_file_names=[".gitkeep", "ci.jld2"],
    #     batch_size=2, n_batch=6, epochs=10,
    #     η₀=1e-2, f_threshold=50, Δf=50,
    #     show_moniter=false
    # )

    # @test length(in_losses) == length(out_losses) == 10

    # rm(joinpath(SqState.training_data_path(), "ci.jld2"))
    # rm(joinpath(SqState.model_path(), "ci.jld2"))
end

@testset "get model" begin
    # m = get_model("model")

    # n = 6
    # gen_squeezed_thermal_data(
    #     n_data=6, n_points=4096,
    #     r_range=(0, 2), θ_range=(0, 2π), n̄_range=(0, 0.5), θ_offset_range=(0, 0),
    #     point_dim=500, label_dim=70,
    #     file_name="ci"
    # )
    # loader = preprocess("ci.jld2", batch_size=2, fragment_size=n)
    # x, y = first(loader)

    # ŷ = m(x)
    # @test size(ŷ) == (3, 2)
    # @test sum((y-ŷ).^2) / 6 < 0.5

    # rm(joinpath(SqState.training_data_path(), "ci.jld2"))
end
