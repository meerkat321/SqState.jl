########
# path #
########

training_data_path() = mkpath(joinpath(datadep"SqState", "training_data"))
model_path() = mkpath(joinpath(datadep"SqState", "model"))
data_path() = mkpath(joinpath(datadep"SqState", "data"))

############
# gen data #
############

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
        c1 * SqueezedState(sq, rep=StateMatrix, dim=dim) +
        c2 * SqueezedThermalState(sq, n̄, dim=dim) +
        c3 * ThermalState(n̄, dim=dim)

    return state
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
