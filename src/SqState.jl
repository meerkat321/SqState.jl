module SqState
    using DataDeps
    using QuantumStateBase

    function __init__()
        register(DataDep("SqState", """Data for SqState.""", ""))
        mkpath(joinpath(DataDeps.standard_loadpath[1], "SqState"))
    end

    # include("plot.jl")
end
