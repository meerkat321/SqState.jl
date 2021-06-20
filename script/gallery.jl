using SqState

function main()
    ########################
    # init wigner function #
    ########################
    x_range = -10:0.02:10
    p_range = -10:0.02:10
    @info "Initialising"
    start_time = time()
    wf = WignerFunction(x_range, p_range)
    end_time = time()
    @info "Done, took $(end_time - start_time)(s)"

    ########
    # plot #
    ########
    PROJECT_PATH = @__DIR__

    @time ws = wf(VacuumState())
    file_path = joinpath(PROJECT_PATH, "../gallery", "wigner_surface_banner.png")
    p = plot_wigner(ws, Surface, size=(1280, 640), file_path=file_path)

    return p
end

main()
