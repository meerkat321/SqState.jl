@testset "training" begin

end

@testset "get model" begin
    m = get_model("model")

    n = 6
    gen_squeezed_thermal_data(
        n_data=6, n_points=4096,
        r_range=(0, 2), θ_range=(0, 2π), n̄_range=(0, 0.5), θ_offset_range=(0, 0),
        point_dim=500, label_dim=70,
        file_name="ci"
    )
    loader = preprocess("ci.jld2", batch_size=2, fragment_size=n)
    x, y = first(loader)

    ŷ = m(x)
    @test size(ŷ) == (3, 2)
    @test sum((y-ŷ).^2) / 6 < 0.5
    
    rm(joinpath(SqState.training_data_path(), "ci.jld2"))
end
