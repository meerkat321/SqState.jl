using Mmap

export
    wigner,
    WignerFunction

#=
    Wigner function by Laguerre Polynominal
=#

const DIM = 35

abstract type CreateWignerMethod end
struct Load𝐖 <: CreateWignerMethod end
struct Calc𝐖 <: CreateWignerMethod end

function wigner(m::Integer, n::Integer, x::Vector{<:Real}, p::Vector{<:Real})
    w = gaussian_function(x, p) .*
        coefficient_of_wave_function(m, n) .*
        z_to_power(m, n, x, p) .*
        laguerre(m, n, x, p)

    return w
end

function create_wigner(
    m_dim::Integer,
    n_dim::Integer,
    x_range::AbstractRange,
    p_range::AbstractRange,
    ::Type{Calc𝐖}
)
    𝐰 = Array{ComplexF64,4}(undef, m_dim, n_dim, length(x_range), length(p_range))
    @sync for m in 1:m_dim
        Threads.@spawn for n in 1:n_dim
            𝐰[m, n, :, :] = wigner(m, n, collect(x_range), collect(p_range))
        end
    end

    return 𝐰
end

function create_wigner(
    m_dim::Integer,
    n_dim::Integer,
    x_range::AbstractRange,
    p_range::AbstractRange,
    bin_path::String,
    ::Type{Load𝐖}
)
    return load_𝐰(m_dim, n_dim, x_range, p_range, bin_path)
end

function create_wigner(
    m_dim::Integer,
    n_dim::Integer,
    x_range::AbstractRange,
    p_range::AbstractRange,
)
    bin_path = gen_wigner_bin_path(m_dim, n_dim, x_range, p_range)

    if isfile(bin_path)
        return create_wigner(m_dim, n_dim, x_range, p_range, bin_path, Load𝐖)
    end

    𝐰 = create_wigner(m_dim, n_dim, x_range, p_range, Calc𝐖)
    # save_𝐰(bin_path, 𝐰)

    return 𝐰
end

mutable struct WignerFunction{T<:Integer, U<:AbstractRange}
    m_dim::T
    n_dim::T
    x_range::U
    p_range::U
    𝐰::Array{ComplexF64,4}

    function WignerFunction(
        m_dim::T,
        n_dim::T,
        x_range::U,
        p_range::U
    ) where {T<:Integer, U<:AbstractRange}
        !check_argv(m_dim, n_dim, x_range, p_range) && throw(ArgumentError)

        𝐰 = create_wigner(m_dim, n_dim, x_range, p_range)
        return new{T, U}(m_dim, n_dim, x_range, p_range, 𝐰)
    end
end

function WignerFunction(x_range::AbstractRange, p_range::AbstractRange; dim=DIM)
    return WignerFunction(dim, dim, x_range, p_range)
end

function (wf::WignerFunction)(ρ::AbstractMatrix)
    𝐰_surface = Matrix{Float64}(undef, length(wf.x_range), length(wf.p_range))
    @sync for i in 1:length(wf.x_range)
        Threads.@spawn for j in 1:length(wf.p_range)
            𝐰_surface[i, j] = real(sum(ρ .* wf.𝐰[:, :, i, j]))
        end
    end

    return 𝐰_surface
end

#########
# utils #
#########

function save_𝐰(bin_path::String, 𝐰::Array{ComplexF64,4})
    @info "Save W_{m,n,x,p} to $bin_path"
    mem = open(bin_path, "w+")
    write(mem, 𝐰)
    close(mem)
end

function load_𝐰(
    m_dim::Integer,
    n_dim::Integer,
    x_range::AbstractRange,
    p_range::AbstractRange,
    bin_path::String
)
    @info "Load W_{m,n,x,p} from $bin_path"
    mem = open(bin_path)
    𝐰 = Mmap.mmap(
        mem,
        Array{ComplexF64,4},
        (m_dim, n_dim, length(x_range), length(p_range))
    )
    close(mem)

    return 𝐰
end

function gen_wigner_bin_path(
    m_dim::Integer,
    n_dim::Integer,
    x_range::AbstractRange,
    p_range::AbstractRange,
)
    path = datadep"SqState"
    bin_path = joinpath(
        path,
        "W " *
        "m=$(m_dim) n=$(n_dim) " *
        "x=$(range2str(x_range)) p=$(range2str(p_range)).bin"
    )

    return bin_path
end

range2str(range::AbstractRange) = replace(string(range), ":" => "_")

check_zero(m_dim, n_dim) = !iszero(m_dim) && !iszero(n_dim)

check_empty(x_range, p_range) = !isempty(x_range) && !isempty(p_range)

function check_argv(m_dim, n_dim, x_range, p_range)
    return check_zero(m_dim, n_dim) && check_empty(x_range, p_range)
end
