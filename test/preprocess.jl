@testset "preprocess for training" begin
    n = 6
    gen_squeezed_thermal_data(
        n_data=6, n_points=4096,
        r_range=(0, 2), θ_range=(0, 2π), n̄_range=(0, 0.5), θ_offset_range=(0, 0),
        point_dim=500, label_dim=70,
        file_name="ci"
    )
    loader = preprocess("ci.jld2", batch_size=2, fragment_size=n)
    x, y = first(loader)

    @test size(x) == (4096, 1, 2)
    @test size(y) == (3, 2)

    rm(joinpath(SqState.training_data_path(), "ci.jld2"))
end
