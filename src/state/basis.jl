export
    FockState,
    NumberState,
    VacuumState,
    SinglePhotonState

function FockState(T::Type{<:Number}, n::Integer, dim::Integer)
    v = zeros(T, dim)
    v[n+1] = 1

    return StateVector{T}(v, dim)
end

FockState(T::Type{<:Number}, n; dim=DIM) = FockState(T, n, dim)

FockState(n; dim=DIM) = FockState(ComplexF64, n, dim)

NumberState(n; dim=DIM) = FockState(ComplexF64, n, dim)

VacuumState(; dim=DIM) = FockState(ComplexF64, 0, dim)

SinglePhotonState(; dim=DIM) = FockState(ComplexF64, 1, dim)
