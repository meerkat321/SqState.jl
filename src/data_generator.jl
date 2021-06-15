using JLD2

export
    pdf_θ_x,
    gen_training_data

tr_mul(𝐚, 𝐛) = sum(𝐚[i, :]' * 𝐛[:, i] for i in 1:size(𝐚, 1))

function pdf_θ_x(state::StateMatrix, θ::Real, x::Real)
    return real(tr_mul(𝛑_θ_x(θ, x, dim=state.dim), state.𝛒))
end

function gen_y(state::StateMatrix; θs = 0:2e-1:2π, xs = -10:5e-1:10)
    pdf = (θ, x) -> pdf_θ_x(state, θ, x)

    𝐩 = Matrix{Float64}(undef, length(θs), length(xs))
    @sync for (i, θ) in enumerate(θs)
        Threads.@spawn for (j, x) in enumerate(xs)
            𝐩[i, j] = pdf(θ, x)
        end
    end

    return 𝐩
end

to_f5(x) = round(x, digits=5)

function gen_training_data(r, θ, n̄; dim=DIM)
    state = SqueezedThermalState(ξ(r, θ), n̄, dim=dim)
    data_path = mkpath(joinpath(datadep"SqState", "training_data", "gen_data"))
    data_name = joinpath(data_path, "$(to_f5(r))_$(to_f5(θ))_$(to_f5(n̄)).jld2")

    p = gen_y(state)
    @time jldsave(data_name; r, θ, n̄, p)
end

function gen_training_data(; rs=0:1e-1:16, θs=0:1e-1:2π, n̄s=0:1e-2:0.5, dim=DIM)
    @info "Start to gen training data" rs θs n̄s
    n_data = length(rs) * length(θs) * length(n̄s)

    i = 0
    for r in rs, θ in θs, n̄ in n̄s
        i += 1
        @info "Args [$i/$n_data]" r θ n̄
        gen_training_data(r, θ, n̄, dim=dim)
    end
end
