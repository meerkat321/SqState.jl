export
    gen_data_sqth_th,
    gen_data_sqth

function gen_data_sqth_th(;
    n_data, n_points=4096,
    r_range=(0., 2.), θ_range=(0., 2π), n̄_range=(0., 0.5),
    point_dim=1000, label_dim=100,
)
    args = Matrix{Float64}(undef, 6, n_data)
    points = Array{Float64, 3}(undef, 2, n_points, n_data)
    𝛒s = Array{ComplexF64, 3}(undef, label_dim, label_dim, n_data)
    σs = Matrix{Float64}(undef, n_points, n_data)

    for i in 1:n_data
        args[:, i] .= r, θ, n̄, c1, c2, c3 = rand_arg_sqth_th(r_range, θ_range, n̄_range)

        # points
        point_dim = (r > 1) ? point_dim : label_dim
        state = construct_state_sqth_th(r, θ, n̄, c1, c2, c3, point_dim)
        _, _, σs[:, i] = gaussian_state_sampler!(view(points, :, :, i), state, 0.)

        # 𝛒s
        𝛒s[:, :, i] = (r > 1) ? state.𝛒[1:label_dim, 1:label_dim] : state.𝛒
    end

    return points, 𝛒s, args, σs
end

function gen_data_sqth(;
    n_data, n_points=4096,
    r_range=(0., 2.), θ_range=(0., 2π), n̄_range=(0., 0.5),
    point_dim=1000, label_dim=100,
)
    args = Matrix{Float64}(undef, 3, n_data)
    points = Array{Float64, 3}(undef, 2, n_points, n_data)
    𝛒s = Array{ComplexF64, 3}(undef, label_dim, label_dim, n_data)
    σs = Matrix{Float64}(undef, n_points, n_data)

    for i in 1:n_data
        args[:, i] .= r, θ, n̄ = rand_arg_sqth(r_range, θ_range, n̄_range)

        # points
        point_dim = (r > 1) ? point_dim : label_dim
        state = construct_state_sqth(r, θ, n̄, point_dim)
        _, _, σs[:, i] = gaussian_state_sampler!(view(points, :, :, i), state, 0.)

        # 𝛒s
        𝛒s[i] = (r > 1) ? state.𝛒[1:label_dim, 1:label_dim] : state.𝛒
    end

    return points, 𝛒s, args, σs
end
