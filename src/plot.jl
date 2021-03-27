using SqState
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

const C_GRAD = cgrad([
    RGBA(53/255, 157/255, 219/255, 1),
    RGBA(240/255, 240/255, 240/255, 1),
    RGBA(219/255, 64/255, 68/255, 1)
])

function plot_wigner(
    wf::WignerFunction, w::AbstractMatrix, ::Type{Heatmap};
    size=(700, 630),
    file_path=nothing
)
    if isnothing(size)
        gr()
    else
        gr(size=size)
    end
    lim = maximum(abs.(w))
    p = Plots.heatmap(
        wf.xs, wf.ps, w,
        title="Wigner Function",
        xlabel="X",
        ylabel="P",
        clim=(-lim, lim),
        c=C_GRAD,
    )

    isnothing(file_path) || savefig(p, file_path)

    return p
end

function plot_wigner(
    wf::WignerFunction, w::AbstractMatrix, ::Type{Contour};
    size=(700, 630),
    file_path=nothing
)
    if isnothing(size)
        gr()
    else
        gr(size=size)
    end
    lim = maximum(abs.(w))
    p = Plots.contour(
        wf.xs, wf.ps, w,
        title="Wigner Function",
        xlabel="X",
        ylabel="P",
        clim=(-lim, lim),
        fill=true,
        levels=20,
        c=C_GRAD,
    )

    isnothing(file_path) || savefig(p, file_path)

    return p
end

function plot_wigner(
    wf::WignerFunction, w::AbstractMatrix, ::Type{Surface};
    size=(700, 630),
    file_path=nothing
)
    if isnothing(size)
        gr()
    else
        gr(size=size)
    end
    lim = maximum(abs.(w))
    p = Plots.surface(
		wf.xs, wf.ps, w,
		title="Wigner Function",
        xlabel="X",
        ylabel="P",
        clim=(-lim, lim),
		zlim=(-lim, lim),
		c=C_GRAD,
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

    if isnothing(size)
        gr()
    else
        gr(size=size)
    end
    lim = maximum(ρᵣ)
    p = Plots.heatmap(
        0:state_n, 0:state_n, ρᵣ,
        title="Density Matrix (Real part)",
        xlabel="m",
        ylabel="n",
        c=C_GRAD,
        clim=(-lim, lim)
    )

    isnothing(file_path) || savefig(p, file_path)

    return p
end

function plot_all(
    wf::WignerFunction, w::AbstractMatrix, ρ::AbstractMatrix;
    state_n=0,
    size=nothing,
    file_path=nothing
)
    if isnothing(size)
        gr()
    else
        gr(size=size)
    end

    l = @layout [
        a{0.6w} grid(2, 1)
    ]

    p = plot(
        plot_wigner(wf, w, Surface, size=nothing),
        plot_wigner(wf, w, Contour, size=nothing),
        plot_ρ(ρ, state_n=state_n, size=nothing),
        layout=l
    )

    return p
end
