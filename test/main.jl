using SqState
using DataDeps

function init_wigner(x_range, p_range)
    @info "Initialising"
    start_time = time()
    wf = WignerFunction(x_range, p_range)
    end_time = time()
    @info "Done, took $(end_time - start_time)(s)"

    return wf
end

function load_data()
    data_path = @datadep_str "SqState/data/dm.h5"
    data_name = "SQ4"
    ρ = read_ρ(data_path, data_name)

    return ρ
end

function render_wigner(wf::WignerFunction, ρ::AbstractMatrix)
    w = wf(ρ)

    return w
end

function plot_data(wf::WignerFunction, w::AbstractMatrix, ρ::AbstractMatrix)
    path = @datadep_str "SqState/render"

    file_path = joinpath(path, "density_matrix_total.png")
    p = plot_ρ(ρ, file_path=file_path)
    file_path = joinpath(path, "density_matrix.png")
    p = plot_ρ(ρ, state_n=5, file_path=file_path)

    file_path = joinpath(path, "wigner_contour.png")
    p = plot_wigner(wf, w, Contour, file_path=file_path)
    file_path = joinpath(path, "wigner_heatmap.png")
    p = plot_wigner(wf, w, Heatmap, file_path=file_path)
    file_path = joinpath(path, "wigner_surface.png")
    p = plot_wigner(wf, w, Surface, file_path=file_path)
    file_path = joinpath(path, "wigner_surface_banner.png")
    p = plot_wigner(wf, w, Surface, size=(1280, 640), file_path=file_path)

    file_path = joinpath(path, "all.png")
    p = plot_all(wf, w, ρ, file_path=file_path)
end

function main()
    x_range = -10:0.1:10
    p_range = -10:0.1:10
    wf = init_wigner(x_range, p_range)
    ρ = load_data()
    w = render_wigner(wf, ρ)

    plot_data(wf, w, ρ)
end

main()
