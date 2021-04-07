module SqState
    using DataDeps

    function __init__()
        register(DataDep(
            "SqState",
            """
            Data for SqState.
            """,
            ""
        ))
    end

    include("state.jl")

    include("read.jl")
    include("utils.jl")
    include("polynomial.jl")
    include("wigner.jl")
    include("plot.jl")
end
