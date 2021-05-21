export
    FockState,
    NumberState,
    VacuumState,
    SinglePhotonState

function FockState(T::Type{<:Number}, n::Integer; dim::Integer)
    v = zeros(T, dim)
    v[n+1] = 1

    return StateVector{T}(v, dim)
end

FockState(n; dim=DIM) = FockState(ComplexF64, n, dim=dim)

NumberState(n; dim=DIM) = FockState(ComplexF64, n, dim=dim)

VacuumState(; dim=DIM) = FockState(ComplexF64, 0, dim=dim)

SinglePhotonState(; dim=DIM) = FockState(ComplexF64, 1, dim=dim)
