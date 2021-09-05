
export post_processing

function merge_l(l_raw, dim)
    b = Int64((dim^2 - dim)/2 + dim)
    l = ComplexF64.(l_raw[1:b])
    for (i, e) in enumerate(l_raw[(b+1):end])
        l[i] += im * e
    end

    return l
end

function reshape_l(l, dim)
    l_ch = zeros(dim, dim)
    start_i = 1
    for i in -(dim-1):0
        l_ch += diagm(i => l[start_i:(start_i+(dim-1)+i)])
        start_i += (dim)+i
    end

    return l_ch
end

function ch2𝛒(l_ch, dim, δ)
    𝛒 = (l_ch' * l_ch) - Matrix{Float64}(I, dim, dim) * δ

    return 𝛒
end

function post_processing(l_raw; dim=100, δ=1e-15)
    return ch2𝛒(reshape_l(merge_l(l_raw, dim), dim), dim, δ)
end
