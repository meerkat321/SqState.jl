export
    fock_state

function fock_state(n::Integer; ρ_size=35)
    # rebase 0-based index system to 1-based
    n += 1

    ρ = zeros(Complex, 35, 35)
    ρ[n, n] = 1

    return ρ
end
