########
# path #
########

training_data_path() = mkpath(joinpath(datadep"SqState", "training_data"))
model_path() = mkpath(joinpath(datadep"SqState", "model"))

############
# gen data #
############

rand2range(x_range) = x_range[1] + (x_range[2]-x_range[1])*rand()

function rand_arg_sqth_th(r_range, θ_range, n̄_range)
    r = rand2range(r_range)
    θ = rand2range(θ_range)
    n̄ = rand2range(n̄_range)
    n̄0 = rand2range(n̄_range)
    c1 = rand()
    c2 = 1 - c1

    return r, θ, n̄, n̄0, c1, c2
end

function rand_arg_sqth(r_range, θ_range, n̄_range)
    r = rand2range(r_range)
    θ = rand2range(θ_range)
    n̄ = rand2range(n̄_range)

    return r, θ, n̄
end

function construct_state_sqth_th(r, θ, n̄, n̄0, c1, c2, dim)
    state =
        c1 * SqueezedThermalState(ξ(r, θ), n̄, dim=dim) +
        c2 * ThermalState(n̄0, dim=dim)

    return state
end

function construct_state_sqth(r, θ, n̄, dim)
    return SqueezedThermalState(ξ(r, θ), n̄, dim=dim)
end

#############
# inference #
#############

function get_model(model_name::String)
    f = jldopen(joinpath(model_path() , "$model_name.jld2"))
    model = f["model"]
    close(f)

    return model
end

infer(m, data::Vector) = m(reshape(Float32.(data), 4096, 1, 1))
