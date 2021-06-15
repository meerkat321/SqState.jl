using JLD2

export
    pdf_θ_x,
    gen_training_data

tr_mul(𝐚, 𝐛) = sum(𝐚[i, :]' * 𝐛[:, i] for i in 1:size(𝐚, 1))

function pdf_θ_x(state::StateMatrix, θ::Real, x::Real)
    return real(tr_mul(𝛑_θ_x(θ, x, dim=state.dim), state.𝛒))
end

function gen_y(state::StateMatrix, θs, xs)
    pdf = (θ, x) -> pdf_θ_x(state, θ, x)

    𝐩 = Matrix{Float64}(undef, length(θs), length(xs))
    @sync for (i, θ) in enumerate(θs)
        Threads.@spawn for (j, x) in enumerate(xs)
            𝐩[i, j] = pdf(θ, x)
        end
    end

    return 𝐩
end

function gen_training_data(;
    rs=0:2e-1:16, θs=0:2e-1:2π, n̄s=0:2.5e-2:0.5,
    bin_θs=0:2e-1:2π, bin_xs=-10:5e-1:10, dim=DIM
)
    data_path = mkpath(joinpath(datadep"SqState", "training_data", "gen_data"))
    data_name = joinpath(data_path, "$dim $(range2str(bin_θs)) $(range2str(bin_θs)).jld2")

    @info "Start to gen training data" rs θs n̄s

    𝐩_dict = Dict{Tuple{Float64, Float64, Float64}, Matrix{Float64}}()
    i = 0
    n_data = length(rs) * length(θs) * length(n̄s)
    for r in rs, θ in θs, n̄ in n̄s
        i += 1
        @info "Args [$i/$n_data]" r θ n̄

        state = SqueezedThermalState(ξ(r, θ), n̄, dim=dim)
        @time 𝐩_dict[(r, θ, n̄)] = gen_y(state, bin_θs, bin_xs)
    end

    jldsave(data_name; 𝐩_dict)
end
