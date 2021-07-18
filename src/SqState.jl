module SqState
    using DataDeps
    using JLD2
    using LinearAlgebra
    using Flux
    using Flux.Data: DataLoader
    using CUDA
    using QuantumStateBase


    function __init__()
        register(DataDep("SqState", """Data for SqState.""", ""))
        mkpath(joinpath(DataDeps.standard_loadpath[1], "SqState"))
    end

    training_data_path() = joinpath(datadep"SqState", "training_data")
    model_path() = joinpath(datadep"SqState", "model")

    # include("plot.jl")
    include("model.jl")
    include("preprocess.jl")
end
