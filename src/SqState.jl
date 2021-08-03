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
    using Transformers.Datasets: download_gdrive

    function __init__()
        register(DataDep(
            "SqState",
            """Data for SqState.""",
            # https://drive.google.com/file/d/1UzaPBpTuhxvmyUWnoOupEr3cRZUMc-0-/view?usp=sharing
            "https://docs.google.com/uc?export=download&id=1UzaPBpTuhxvmyUWnoOupEr3cRZUMc-0-",
            "8bc0c64d09b17c92df4c2065064ae09edb0ea56b05bdc196a0e7d21a998e1fea";
            fetch_method=download_gdrive,
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

    include("real_time_system.jl")
end
