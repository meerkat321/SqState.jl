using LinearAlgebra

export
    CoherentState,
    SqueezedState

include("representation.jl")
include("basis.jl")
include("operator.jl")

function CoherentState(arg::Arg{<:Real}; dim=DIM)
    return displace!(VacuumState(dim=dim), arg)
end

function SqueezedState(ξ::Arg{<:Real}; dim=DIM)
    return squeeze!(VacuumState(dim=dim), ξ)
end

bose_einstein(n::Integer, n̄::Real) = n̄^n / (1 + n̄)^(n+1)

bose_einstein(n̄::Real) = n -> bose_einstein(n, n̄)

ThermalState(n̄::Real; dim=DIM) = StateMatrix(diagm(bose_einstein(n̄).(0:dim-1)), dim)

function SqueezedThermalState(ξ::Arg{<:Real}, n̄::Real; dim=DIM)
    return squeeze!(ThermalState(n̄, dim=dim), ξ)
end
