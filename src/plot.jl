using Plots

export
    Heatmap,
    Contour,
    Surface,
    plot_wigner,
    plot_ρ,
    plot_all

abstract type PlotMethod end

struct Heatmap <: PlotMethod end

struct Contour <: PlotMethod end

struct Surface <: PlotMethod end

function plot_wigner(
    wf::WignerFunction, w::AbstractMatrix, ::Type{Heatmap};
    size=(700, 630),
    file_path=nothing
)
    !isnothing(size) && (gr(size=size) isa Plots.GRBackend) || gr()

    lim = maximum(abs.(w))
    p = Plots.heatmap(
        wf.x_range, wf.p_range, w,
        title="Wigner Function",
        xlabel="X",
        ylabel="P",
        clim=(-lim, lim),
        c=:coolwarm,
    )

    isnothing(file_path) || savefig(p, file_path)

    return p
end

function plot_wigner(
    wf::WignerFunction, w::AbstractMatrix, ::Type{Contour};
    size=(700, 630),
    file_path=nothing
)
    !isnothing(size) && (gr(size=size) isa Plots.GRBackend) || gr()

    lim = maximum(abs.(w))
    p = Plots.contour(
        wf.x_range, wf.p_range, w,
        title="Wigner Function",
        xlabel="X",
        ylabel="P",
        clim=(-lim, lim),
        fill=true,
        levels=20,
        c=:coolwarm,
    )

    isnothing(file_path) || savefig(p, file_path)

    return p
end

function plot_wigner(
    wf::WignerFunction, w::AbstractMatrix, ::Type{Surface};
    size=(700, 630),
    file_path=nothing
)
    !isnothing(size) && (gr(size=size) isa Plots.GRBackend) || gr()

    lim = maximum(abs.(w))
    p = Plots.surface(
        wf.x_range, wf.p_range, w,
        title="Wigner Function",
        xlabel="X",
        ylabel="P",
        clim=(-lim, lim),
        zlim=(-lim, lim),
        c=:coolwarm,
        fillalpha=0.99,
        camera=(40, 30),
    )

    isnothing(file_path) || savefig(p, file_path)

    return p
end

function plot_ρ(
    ρ::AbstractMatrix;
    state_n=0,
    size=(700, 630),
    file_path=nothing
)
    ρᵣ = real(ρ)
    if state_n != 0
        ρᵣ = ρᵣ[1:state_n+1, 1:state_n+1]
    else
        state_n = Base.size(ρᵣ)[1] - 1
    end

    !isnothing(size) && (gr(size=size) isa Plots.GRBackend) || gr()

    lim = maximum(ρᵣ)
    p = Plots.heatmap(
        0:state_n, 0:state_n, ρᵣ,
        title="Density Matrix (Real part)",
        xlabel="m",
        ylabel="n",
        c=:coolwarm,
        clim=(-lim, lim)
    )

    isnothing(file_path) || savefig(p, file_path)

    return p
end

function plot_all(
    wf::WignerFunction, w::AbstractMatrix, ρ::AbstractMatrix;
    state_n=0,
    file_path=nothing
)
    gr()
    l = @layout [a{0.6w} grid(2, 1)]
    p = plot(
        plot_wigner(wf, w, Surface, size=nothing),
        plot_wigner(wf, w, Contour, size=nothing),
        plot_ρ(ρ, state_n=state_n, size=nothing),
        layout=l
    )

    isnothing(file_path) || savefig(p, file_path)

    return p
end
