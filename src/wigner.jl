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

wigner(m::Integer, n::Integer) = (x::Real, p::Real)->wigner(m, n, x, p)

function create_wigner(
    m_dim::Integer,
    n_dim::Integer,
    x_range::AbstractRange,
    p_range::AbstractRange
)
    𝐰 = Array{ComplexF64,4}(undef, m_dim, n_dim, length(x_range), length(p_range))
    @sync for m in 1:m_dim
        for n in 1:n_dim
            for (x_i, x) in enumerate(x_range)
                Threads.@spawn for (p_j, p) in enumerate(p_range)
                    𝐰[m, n, x_i, p_j] = wigner(m ,n, x, p)
                end
            end
        end
    end

    path = datadep"SqState"
    bin_path = joinpath(
        path,
        "W m=$(m_dim) n=$(n_dim) x=$(replace(string(x_range), ":" => "_")) p=$(replace(string(p_range), ":" => "_")).bin"
    )
    save_𝐰(bin_path, 𝐰)

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

        # load from Mmap
        path = datadep"SqState"
        bin_path = joinpath(
            path,
            "W m=$(m_dim) n=$(n_dim) x=$(replace(string(x_range), ":" => "_")) p=$(replace(string(p_range), ":" => "_")).bin"
        )
        if isfile(bin_path)
            𝐰 = load_𝐰(m_dim, n_dim, x_range, p_range, bin_path)
            return new{T, U}(m_dim, n_dim, x_range, p_range, 𝐰)
        end

        𝐰 = create_wigner(m_dim, n_dim, x_range, p_range)
        return new{T, U}(m_dim, n_dim, x_range, p_range, 𝐰)
    end
end

function WignerFunction(x_range::AbstractRange, p_range::AbstractRange; dim=35)
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

check_zero(m_dim, n_dim) = !iszero(m_dim) && !iszero(n_dim)

check_empty(x_range, p_range) = !isempty(x_range) && !isempty(p_range)

function check_argv(m_dim, n_dim, x_range, p_range)
    return check_zero(m_dim, n_dim) && check_empty(x_range, p_range)
end
