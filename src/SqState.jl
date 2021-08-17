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
    using Fetch
    using MAT

    function __init__()
        register(DataDep(
            "SqState",
            """Data and models for SqState.""",
            "https://drive.google.com/file/d/1UzaPBpTuhxvmyUWnoOupEr3cRZUMc-0-/view?usp=sharing",
            "8bc0c64d09b17c92df4c2065064ae09edb0ea56b05bdc196a0e7d21a998e1fea";
            fetch_method=gdownload,
            post_fetch_method=unpack
        ))
    end

    training_data_path() = joinpath(datadep"SqState", "training_data")
    model_path() = joinpath(datadep"SqState", "model")
    data_path() = joinpath(datadep"SqState", "data")

    include("gen_data.jl")
    include("preprocess.jl")
    include("model.jl")
    include("training.jl")

    include("real_time_system.jl")
end
