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

function create_wigner(m_dim::Integer, n_dim::Integer, xs::AbstractRange, ps::AbstractRange)
    W = Array{ComplexF64,4}(undef, m_dim, n_dim, length(xs), length(ps))
    @sync for m in 1:m_dim
        for n in 1:n_dim
            for (x_i, x) in enumerate(xs)
                Threads.@spawn for (p_j, p) in enumerate(ps)
                    W[m, n, x_i, p_j] = wigner(m ,n, x, p)
                end
            end
        end
    end

    path = @datadep_str "SqState"
    bin_path = joinpath(path, "W_$(m_dim)_$(n_dim)_$(xs)_$(ps).bin")
    @info "Save W_{m,n,x,p} to $bin_path"
    mem = open(bin_path, "w+")
    write(mem, W)
    close(mem)

    return W
end

mutable struct WignerFunction{T<:Integer}
    m_dim::T
    n_dim::T
    xs
    ps
    W::Array{ComplexF64,4}

    function WignerFunction(m_dim::T, n_dim::T, xs::AbstractRange, ps::AbstractRange) where {T<:Integer}
        path = @datadep_str "SqState"
        bin_path = joinpath(path, "W_$(m_dim)_$(n_dim)_$(xs)_$(ps).bin")
        if isfile(bin_path)
            @info "Load W_{m,n,x,p} from $bin_path"
            mem = open(bin_path)
            W = Mmap.mmap(mem, Array{ComplexF64,4}, (m_dim, n_dim, length(xs), length(ps)))

            return new{T}(m_dim, n_dim, xs, ps, W)
        end

        if check_zero(m_dim, n_dim) && check_empty(xs, ps)
            W = create_wigner(m_dim, n_dim, xs, ps)
        else
            W = Array{ComplexF64,4}(undef, 0, 0, 0, 0)
        end

        return new{T}(m_dim, n_dim, xs, ps, W)
    end
end

function WignerFunction(xs::AbstractRange, ps::AbstractRange; dim=35)
    return WignerFunction(dim, dim, xs, ps)
end

function (wf::WignerFunction)(ρ::AbstractMatrix)
    reshape(real(sum(ρ .* wf.W, dims=(1, 2))), length(wf.xs), length(wf.ps))
end

function Base.setproperty!(wf::WignerFunction, name::Symbol, x)
    setfield!(wf, name, x)
    m_dim = getproperty(wf, :m_dim)
    n_dim = getproperty(wf, :n_dim)
    xs = getproperty(wf, :xs)
    ps = getproperty(wf, :ps)
    if check_zero(m_dim, n_dim) && check_empty(xs, ps)
        W = create_wigner(m_dim, n_dim, xs, ps)
        setfield!(wf, :W, W)
    end
end

check_zero(m_dim, n_dim) = !iszero(m_dim) && !iszero(n_dim)

check_empty(xs, ps) = !isempty(xs) && !isempty(ps)
