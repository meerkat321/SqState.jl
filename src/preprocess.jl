using MAT

export
    preprocess,
    infer

############
# training #
############

function preprocess(file_name::String; batch_size=50, fragment_size=10000)
    f = jldopen(joinpath(SqState.training_data_path(), file_name), "r")
    points = f["points"][2, :, :]

    # 4096 points 1 channel, 10000 data in a data fragment
    xs = reshape(Float32.(points), (4096, 1, fragment_size))

    # (r, θ, n̄), 10000 data in data fragment
    ys = f["args"][1:3, :]

    close(f)

    return DataLoader((xs, ys), batchsize=batch_size, shuffle=true)
end

#############
# inference #
#############

function get_data(data_name::String)
    data_file = matopen(joinpath(data_path(), "Flow/$data_name"))
    data = read(data_file, "data_sq")
    close(data_file)

    return data
end

function sample(data::Matrix, n::Integer)
    data_indices = sort!(rand(1:size(data, 1), n))

    return data[data_indices, 1] # 1: x; 2: θ
end

function infer_arg(data::Matrix, n_sample::Integer; m=get_model("model"))
    argv = zeros(Float32, 3)
    for _ in 1:n_sample
        argv += m(reshape(Float32.(sample(data, 4096)), (4096, 1, 1)))
    end

    return argv / n_sample
end

function calc_w(
    r::T, θ::T, n̄::T, dim::Integer, fix_θ::Bool;
    wf=WignerFunction(LinRange(-3, 3, 100), LinRange(-3, 3, 100), dim=dim)
) where {T<:Real}
    θ = fix_θ ? zero(T) : θ
    state = SqueezedThermalState(ξ(r, θ), n̄, dim=dim)
    w = wf(state)

    return state, w
end

reshape_infered_data(data::Matrix) = [data[:, i] for i in 1:size(data, 2)]

function infer(data_name::String; n_sample=10, fix_θ=true, dim=70)
    data = get_data(data_name)
    state, w = calc_w(infer_arg(data, n_sample)..., dim, fix_θ)

    return reshape_infered_data(real.(𝛒(state))), reshape_infered_data(w.𝐰_surface)
end
