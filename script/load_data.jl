using SqState
using JLD2

file_names = readdir(SqState.training_data_path())

batch = Channel(5, spawn=true) do ch
    for (i, file_name) in enumerate(file_names)
        f = jldopen(joinpath(SqState.training_data_path(), file_name), "r")
        points = f["points"]
        𝛒s = f["𝛒s"]
        put!(batch, (i, points, 𝛒s))

        @info "Loaded $i batch into buffer"
    end
end

function test_ch()
    for (i, x, y) in batch
        println(i, size(x), size(y))
        sleep(1)
    end
end

test_ch()
