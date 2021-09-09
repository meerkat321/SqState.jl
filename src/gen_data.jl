export gen_data

function gen_data(;
    n_data, n_points=4096,
    r_range=(0., 2.), θ_range=(0., 2π), n̄_range=(0., 1.),
    point_dim=1000, label_dim=100,
)
    args = Matrix{Float64}(undef, 6, n_data)
    points = Array{Float64, 3}(undef, 2, n_points, n_data)
    𝛒s = Vector{Matrix{ComplexF64}}(undef, n_data)
    σs = Matrix{Float64}(undef, n_points, n_data)

    for i in 1:n_data
        args[:, i] .= r, θ, n̄, c1, c2, c3 = rand_arg(r_range, θ_range, n̄_range)

        # points
        point_dim = (r > 1) ? point_dim : label_dim
        state = construct_state(r, θ, n̄, c1, c2, c3, point_dim)
        _, _, σs[ :, i] = gaussian_state_sampler!(view(points, :, :, i), state, 0.)

        # 𝛒s
        𝛒s[i] = (r > 1) ? state.𝛒[1:label_dim, 1:label_dim] : state.𝛒
    end

    return points, 𝛒s, args, σs
end
