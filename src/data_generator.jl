using JLD2

export
    pdf_θ_x,
    gen_training_data

real_tr_mul(𝐚, 𝐛) = sum(real(𝐚[i, :]' * 𝐛[:, i]) for i in 1:size(𝐚, 1))

function pdf_θ_x(state::StateMatrix, θ::Real, x::Real)
    return real_tr_mul(𝛑_θ_x(θ, x, dim=state.dim), state.𝛒)
end

function calc_p!(state::StateMatrix, θs, xs, 𝐩::Matrix)
    sp_lock = Threads.SpinLock()
    @sync for (i, θ) in enumerate(θs)
        Threads.@spawn for (j, x) in enumerate(xs)
            p = pdf_θ_x(state, θ, x)
            lock(sp_lock) do
                𝐩[i, j] = p
            end
        end
    end
end

function rand_arg(
    r_range::Tuple{Float64, Float64},
    θ_range::Tuple{Float64, Float64},
    n̄_range::Tuple{Float64, Float64}
)
    r = r_range[1] + (r_range[2]-r_range[1])*rand()
    θ = θ_range[1] + (θ_range[2]-θ_range[1])*rand()
    n̄ = n̄_range[1] + (n̄_range[2]-n̄_range[1])*rand()

    return r, θ, n̄
end

function gen_training_data(
    n;
    r_range=(0., 16.), θ_range=(0., 2π), n̄_range=(0., 0.5),
    bin_θs=0:2e-1:2π, bin_xs=-10:5e-1:10, dim=DIM
)
    data_path = mkpath(joinpath(datadep"SqState", "training_data", "gen_data"))
    data_name = joinpath(data_path, "$dim $(range2str(bin_θs)) $(range2str(bin_θs)).jld2")

    @info "Start to gen training data" r_range θ_range n̄_range

    𝐩_dict = Dict([
        rand_arg(r_range, θ_range, n̄_range)=>Matrix{Float64}(undef, length(bin_θs), length(bin_xs))
        for _ in 1:n
    ])
    for (i, data) in enumerate(𝐩_dict)
        r, θ, n̄ = data.first
        @info "Args [$i/$n]" r θ n̄

        state = SqueezedThermalState(ξ(r, θ), n̄, dim=dim)
        @time calc_p!(state, bin_θs, bin_xs, data.second)
    end

    jldsave(data_name; 𝐩_dict)
end
