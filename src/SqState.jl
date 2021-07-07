module SqState
    using DataDeps
    using QuantumStateBase

    function __init__()
        register(DataDep("SqState", """Data for SqState.""", ""))
        mkpath(joinpath(DataDeps.standard_loadpath[1], "SqState"))
    end

    training_data_path() = joinpath(datadep"SqState", "training_data")

    # include("plot.jl")
end
