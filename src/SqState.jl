module SqState
    using Dates
    using DataDeps
    using JLD2
    using Flux
    using Flux.Data: DataLoader
    using CUDA
    using QuantumStateBase
    using Fetch
    using MAT
    using NeuralOperators

    function __init__()
        register(DataDep(
            "SqState",
            """Data and models for SqState.""",
            "https://drive.google.com/file/d/1UzaPBpTuhxvmyUWnoOupEr3cRZUMc-0-/view?usp=sharing",
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
    include("postprocess.jl")
end
