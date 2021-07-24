export
    gen_squeezed_thermal_data,
    gen_non_gaussian_data

rand2range(x_range) = x_range[1] + (x_range[2]-x_range[1])*rand()

function rand_arg(r_range, θ_range, n̄_range, θ_offset_range)
    r = rand2range(r_range)
    θ = rand2range(θ_range)
    n̄ = rand2range(n̄_range)
    # θ_offset = rand2range(θ_offset_range)
    θ_offset = 0.

    return r, θ, n̄, θ_offset
end

function gen_squeezed_thermal_data(;
    n_data, n_points=4096,
    r_range=(0, 2), θ_range=(0, 2π), n̄_range=(0, 0.5), θ_offset_range=(0, 2π),
    point_dim=500, label_dim=70,
    file_name="gaussian_$(replace(string(now()), ':'=>'_'))"
)
    args = Matrix{Float64}(undef, 4, n_data)
    points = Array{Float64, 3}(undef, 2, n_points, n_data)
    𝛒s = Vector{Matrix{ComplexF64}}(undef, n_data)

    for i in 1:n_data
        args[:, i] .= rand_arg(r_range, θ_range, n̄_range, θ_offset_range)
        r, θ, n̄, θ_offset = args[:, i]

        # points
        state = SqueezedThermalState(ξ(r, θ), n̄, dim=point_dim)
        gaussian_state_sampler!(view(points, :, :, i), state, θ_offset)

        # 𝛒s
        𝛒s[i] = SqueezedThermalState(ξ(r, θ), n̄, dim=label_dim).𝛒
    end

    isnothing(file_name) && return
    data_path = mkpath(SqState.training_data_path())
    jldsave(joinpath(data_path, "$file_name.jld2");
        points, 𝛒s, args,
        n_data, n_points,
        r_range, θ_range, n̄_range, θ_offset_range,
        point_dim, label_dim
    )
end

function gen_non_gaussian_data()
    # QuantumStateBase.extend_coeff_ψₙ!(500)
end
