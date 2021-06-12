using SqState
using DataDeps
using HDF5
using Plots

function gen_training_data()

end

function main()
    path = mkpath(joinpath(datadep"SqState", "training_data"))
    file_name = joinpath(path, "training_data.h5")

    # TODO: DEBUG
    r = 2.
    θ = π/4
    n̄ = 0.3

    sp_lock = Threads.SpinLock()
    # for r in LinRange(0., 16., 35)
    #     for θ in LinRange(0., 2π, 60)
    #         for n̄ in LinRange(0., 0.5, 50)
                data_name = "$(round(r, digits=5)),$(round(θ, digits=5)),$(round(n̄, digits=5))"
                data = gen_data(CoherentState(α(r, θ), rep=StateMatrix), n=4096)

                lock(sp_lock) do
                    h5write(file_name, "$data_name", data)
                end
    #         end
    #     end
    # end

    scatter(data[:, 1], data[:, 2])
end
