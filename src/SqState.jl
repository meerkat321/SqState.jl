module SqState
    using QuantumStateBase

    using Fetch
    using DataDeps

    using CUDA
    using Flux
    using Flux.Data: DataLoader
    using JLD2

    using NeuralOperators

    function __init__()
        register(DataDep(
            "SqState",
            """Data and models for SqState.""",
            "https://drive.google.com/file/d/1UzaPBpTuhxvmyUWnoOupEr3cRZUMc-0-/view?usp=sharing",
            "d867cf78dfb0497f372fff29f76426b1dec07ee02147a3b7d59c4f040ed4a04d",
            fetch_method=gdownload,
            post_fetch_method=unpack
        ))
    end

    include("utils.jl")
    include("gen_data.jl")
    include("preprocess.jl")
    include("model.jl")
    include("training.jl")
    include("postprocess.jl")
end
