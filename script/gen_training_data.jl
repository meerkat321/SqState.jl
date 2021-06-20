using SqState
using DataDeps
using JLD2

rand2range(x_range) = x_range[1] + (x_range[2]-x_range[1])*rand()

function rand_arg(r_range, θ_range, n̄_range, bias_phase_range)
    r = rand2range(r_range)
    θ = rand2range(θ_range)
    n̄ = rand2range(n̄_range)
    bias_phase = rand2range(bias_phase_range)

    return r, θ, n̄, bias_phase
end

function start(;
    n_data, n_points=40960,
    r_range=(0, 2), θ_range=(0, π/2), n̄_range=(0, 0.5), bias_phase_range=(0, 2π),
    point_dim=500, label_dim=70,
    file_name="$(time_ns())"
)
    args = Matrix{Float64}(undef, 4, n_data)
    points = Matrix{Float64}(undef, n_points, n_data)
    𝛒s = Vector{Matrix{ComplexF64}}(undef, n_data)

    for i in 1:n_data
        view(args, :, i) .= rand_arg(r_range, θ_range, n̄_range, bias_phase_range)
        r, θ, n̄, bias_phase = view(args, :, i)

        # points
        state = SqueezedThermalState(ξ(r, θ), n̄, dim=point_dim)
        gen_gaussian_training_data!(view(points, :, i), state, bias_phase)

        # 𝛒s
        𝛒s[i] = SqueezedThermalState(ξ(r, θ), n̄, dim=label_dim).𝛒
    end

    isnothing(file_name) && return
    data_path = mkpath(joinpath(datadep"SqState", "training_data"))
    jldsave(joinpath(data_path, "$file_name.jld2");
        points, 𝛒s, args,
        n_data, n_points,
        r_range, θ_range, n̄_range, bias_phase_range,
        point_dim, label_dim
    )
end

# jit
@time start(n_data=5, file_name=nothing)

# generate training data
@time start(n_data=500)
