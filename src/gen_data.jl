export
    gen_squeezed_thermal_data,
    gen_non_gaussian_data

rand2range(x_range) = x_range[1] + (x_range[2]-x_range[1])*rand()

function rand_arg(r_range, θ_range, n̄_range, θ_offset_range)
    r = rand2range(r_range)
    θ = rand2range(θ_range)
    n̄ = rand2range(n̄_range)
    θ_offset = rand2range(θ_offset_range)

    return r, θ, n̄, θ_offset
end

function gen_squeezed_thermal_data(;
    n_data, n_points=4096,
    r_range=(0, 2), θ_range=(0, 2π), n̄_range=(0, 0.5), θ_offset_range=(0, 0),
    point_dim=900, label_dim=100,
    file_name="gaussian_$(replace(string(now()), ':'=>'_'))"
)
    args = Matrix{Float64}(undef, 4, n_data)
    points = Array{Float64, 3}(undef, 2, n_points, n_data)
    𝛒s = Vector{Matrix{ComplexF64}}(undef, n_data)

    for i in 1:n_data
        args[:, i] .= rand_arg(r_range, θ_range, n̄_range, θ_offset_range)
        r, θ, n̄, θ_offset = args[:, i]

        point_dim = r ≤ 1 ? 100 : point_dim

        # points
        state = SqueezedThermalState(ξ(r, θ), n̄, dim=point_dim)
        gaussian_state_sampler!(view(points, :, :, i), state, θ_offset)

        # 𝛒s
        𝛒s[i] = r ≤ 1 ? state.𝛒 : SqueezedThermalState(ξ(r, θ), n̄, dim=label_dim).𝛒
    end

    if !isnothing(file_name)
        data_path = mkpath(SqState.training_data_path())
        jldsave(
            joinpath(data_path, "$file_name.jld2");
            points, 𝛒s, args,
            n_data, n_points,
            r_range, θ_range, n̄_range, θ_offset_range,
            point_dim, label_dim
        )

        file = matopen(joinpath(data_path, "$file_name.mat"), "w")
        write(file, "points", points)
        write(file, "dms", 𝛒s)
        write(file, "args", args)
        write(file, "n_data", n_data)
        write(file, "n_points", n_points)
        close(file)
    end

    return points, 𝛒s, args,
        n_data, n_points,
        r_range, θ_range, n̄_range, θ_offset_range,
        point_dim, label_dim
end

function gen_non_gaussian_data()
    # QuantumStateBase.extend_coeff_ψₙ!(500)
end
