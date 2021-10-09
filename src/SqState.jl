module SqState
    using QuantumStateBase
    using LinearAlgebra

    using Fetch, DataDeps, JLD2

    using CUDA
    using Flux
    using Flux.Data: DataLoader
    using Zygote, ChainRulesCore


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

    include("utils.jl")
    include("gen_data.jl")
    include("preprocess.jl")
    include("model.jl")
end
