using JLD, HDF5
include("Code.jl")

TIME_LIMIT = 1800
data = Vector{Vector{String}}();
println(data)


for J in [6, 8, 10]
    for L in [2]
        for T in [3]
            for index in 1:5
                for SJ in 0:J
                    filenm = join(["Instances/Small-sized/Ins_", string(J), "_6_", string(L), "_", string(T), "_", string(index), "_", string(SJ), ".jld"])
                    if isfile(filenm)

                        println(filenm, "\n")
                        global_opt_val, global_opt_LB, global_opt_gap, time = @time MILP(filenm, TIME_LIMIT)
                        tempData = Vector{String}(undef, 5)
                        tempData[1] = join(["I_", string(J), "_6_", string(L), "_", string(T), "_", string(index), "_", string(SJ), ".jld"])
                        tempData[2] = string(round(global_opt_val))
                        tempData[3] = string(round(global_opt_LB))
                        tempData[4] = string(round(global_opt_gap * 100, digits=2))
                        tempData[5] = string(round(time, digits=2))
                        push!(data, tempData)
                        println(data)
                    end
                end
            end
        end
    end
end

println(data)
