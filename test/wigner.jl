@testset "wigner" begin
    m, n, x, p = 40, 3, -9.1, 10.5
    @test wigner(m, n, x, p) ==
        SqState.gaussian_function(x, p) *
        SqState.coefficient_of_wave_function(m, n) *
        SqState.z_to_power(m, n, x, p) *
        laguerre(m, n, x, p)
end

@testset "WignerFunction" begin
    m_dim = 10
    n_dim = 10
    xs = -1:0.1:1
    ps = -1:0.1:1

    # w/o mmap
    for _ in 1:2
        wf = WignerFunction(xs, ps)
        ρ = ones(ComplexF64, 35, 35)
        w = wf(ρ)
        ans = real(sum(ρ .* wf.𝐰, dims=(1, 2)))
        for (i, e) in enumerate(w)
            @test e == ans[i]
        end

        wf = WignerFunction(xs, ps, dim=m_dim)
        @test size(wf.𝐰) == (m_dim, n_dim, length(xs), length(ps))
    end

end
