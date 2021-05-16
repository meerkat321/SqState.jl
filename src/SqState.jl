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

        mkpath(joinpath(DataDeps.standard_loadpath[1], "SqState"))
    end

    include("state.jl")

    include("read.jl")
    include("polynomial.jl")
    include("utils.jl")
    include("wigner.jl")
    include("plot.jl")
end
