using SqState

function main()
    ########################
    # init wigner function #
    ########################
    # x_range = -10:0.1:10
    # p_range = -10:0.1:10
    # @info "Initialising"
    # start_time = time()
    # wf = WignerFunction(x_range, p_range)
    # end_time = time()
    # @info "Done, took $(end_time - start_time)(s)"

    ########
    # plot #
    ########
    w = wf(ρ(VacuumState()))
    file_path = joinpath(SqState.PROJECT_PATH, "../gallery", "wigner_surface_banner.png")
    p = plot_wigner(wf, w, Surface, size=(1280, 640), file_path=file_path)

    w = wf(ρ(VacuumState()))
    file_path = joinpath(SqState.PROJECT_PATH, "../gallery", "wigner_surface_|0>.png")
    p = plot_wigner(wf, w, Surface, size=(640, 320), file_path=file_path)

    w = wf(ρ(SinglePhotonState()))
    file_path = joinpath(SqState.PROJECT_PATH, "../gallery", "wigner_surface_|1>.png")
    p = plot_wigner(wf, w, Surface, size=(640, 320), file_path=file_path)
    file_path = joinpath(SqState.PROJECT_PATH, "../gallery", "wigner_contour_|1>.png")
    p = plot_wigner(wf, w, Contour, size=(640, 600), file_path=file_path)

    return p
end

main()
