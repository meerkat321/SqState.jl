module SqState
    using Dates
    using DataDeps
    using JLD2
    using LinearAlgebra
    using Flux
    using Flux.Data: DataLoader
    using CUDA
    using QuantumStateBase
    using UnicodePlots


    function __init__()
        register(DataDep("SqState", """Data for SqState.""", ""))
        mkpath(joinpath(DataDeps.standard_loadpath[1], "SqState"))
    end

    training_data_path() = joinpath(datadep"SqState", "training_data")
    model_path() = joinpath(datadep"SqState", "model")
    data_path() = joinpath(datadep"SqState", "data")

    include("gen_data.jl")
    include("preprocess.jl")
    include("model.jl")
    include("training.jl")
    include("postprocess.jl")

    include("real_time_system.jl")
end
