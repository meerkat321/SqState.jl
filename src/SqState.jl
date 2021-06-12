module SqState
    using DataDeps

    const DIM = 70

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

    include("state/state.jl")

    include("read.jl")
    include("polynomial.jl")
    include("wigner.jl")
    include("plot.jl")
end
