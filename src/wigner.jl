using Mmap

export
    wigner,
    WignerFunction

#=
    Wigner function by Laguerre Polynominal
=#

function wigner(m::Integer, n::Integer, x::Real, p::Real)
    w = gaussian_function(x, p)
    w *= coefficient_of_wave_function(m, n)
    w *= z_to_power(m, n, x, p)
    w *= laguerre(m, n, x, p)

    return w
end

wigner(m::Integer, n::Integer) = (x, p)->wigner(m, n, x, p)

function create_wigner(m_dim::Integer, n_dim::Integer, x_range::AbstractRange, p_range::AbstractRange)
    W = Array{ComplexF64,4}(undef, m_dim, n_dim, length(x_range), length(p_range))
    @sync for m in 1:m_dim
        for n in 1:n_dim
            for (x_i, x) in enumerate(x_range)
                Threads.@spawn for (p_j, p) in enumerate(p_range)
                    W[m, n, x_i, p_j] = wigner(m ,n, x, p)
                end
            end
        end
    end

    path = @datadep_str "SqState"
    bin_path = joinpath(path, "W_$(m_dim)_$(n_dim)_$(x_range)_$(p_range).bin")
    @info "Save W_{m,n,x,p} to $bin_path"
    mem = open(bin_path, "w+")
    write(mem, W)
    close(mem)

    return W
end

mutable struct WignerFunction{T<:Integer}
    m_dim::T
    n_dim::T
    x_range
    p_range
    W::Array{ComplexF64,4}

    function WignerFunction(m_dim::T, n_dim::T, x_range::AbstractRange, p_range::AbstractRange) where {T<:Integer}
        path = @datadep_str "SqState"
        bin_path = joinpath(path, "W_$(m_dim)_$(n_dim)_$(x_range)_$(p_range).bin")
        if isfile(bin_path)
            @info "Load W_{m,n,x,p} from $bin_path"
            mem = open(bin_path)
            W = Mmap.mmap(mem, Array{ComplexF64,4}, (m_dim, n_dim, length(x_range), length(p_range)))

            return new{T}(m_dim, n_dim, x_range, p_range, W)
        end

        if check_zero(m_dim, n_dim) && check_empty(x_range, p_range)
            W = create_wigner(m_dim, n_dim, x_range, p_range)
        else
            W = Array{ComplexF64,4}(undef, 0, 0, 0, 0)
        end

        return new{T}(m_dim, n_dim, x_range, p_range, W)
    end
end

function WignerFunction(x_range::AbstractRange, p_range::AbstractRange; dim=35)
    return WignerFunction(dim, dim, x_range, p_range)
end

function (wf::WignerFunction)(ρ::AbstractMatrix)
    reshape(real(sum(ρ .* wf.W, dims=(1, 2))), length(wf.x_range), length(wf.p_range))
end

check_zero(m_dim, n_dim) = !iszero(m_dim) && !iszero(n_dim)

check_empty(x_range, p_range) = !isempty(x_range) && !isempty(p_range)
