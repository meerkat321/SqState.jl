export
    gen_data,
    gen_non_gaussian_data

rand2range(x_range) = x_range[1] + (x_range[2]-x_range[1])*rand()

function rand_arg(r_range, θ_range, n̄_range)
    r = rand2range(r_range)
    θ = rand2range(θ_range)
    n̄ = rand2range(n̄_range)
    c1 = rand()
    c2 = (1 - c1) * rand()
    c3 = 1 - c1 - c2

    return r, θ, n̄, c1, c2, c3
end

function construct_state(r, θ, n̄, c1, c2, c3, dim)
    sq = ξ(r, θ)
    state =
        c1 * SqueezedState(sq, dim=dim, rep=StateMatrix) +
        c2 * SqueezedThermalState(sq, n̄, dim=dim) +
        c3 * ThermalState(n̄, dim=dim)

    return state
end

function gen_data(;
    n_data, n_points=4096,
    r_range=(0, 2), θ_range=(0, 2π), n̄_range=(0, 1.),
    point_dim=1000, label_dim=100,
    file_name="sq_sqth_th_$(replace(string(now()), ':'=>'_'))"
)
    args = Matrix{Float64}(undef, 6, n_data)
    points = Array{Float64, 3}(undef, 2, n_points, n_data)
    𝛒s = Vector{Matrix{ComplexF64}}(undef, n_data)

    for i in 1:n_data
        args[:, i] .= r, θ, n̄, c1, c2, c3 = rand_arg(r_range, θ_range, n̄_range)

        # points
        point_dim = (r > 1) ? point_dim : label_dim
        state = construct_state(r, θ, n̄, c1, c2, c3, point_dim)
        gaussian_state_sampler!(view(points, :, :, i), state, 0.)

        # 𝛒s
        𝛒s[i] = (r > 1) ? state.𝛒[1:label_dim, 1:label_dim] : state.𝛒
    end

    if !isnothing(file_name)
        data_path = mkpath(SqState.training_data_path())
        jldsave(joinpath(data_path, "$file_name.jld2"); points, 𝛒s, args)

        file = matopen(joinpath(data_path, "../mat_data/$file_name.mat"), "w")
        write(file, "points", points); write(file, "dms", 𝛒s); write(file, "args", args)
        close(file)
    end

    return points, 𝛒s, args
end

function gen_non_gaussian_data()
    # QuantumStateBase.extend_coeff_ψₙ!(500)
end
