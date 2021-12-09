using DelimitedFiles,LuxurySparse,LinearAlgebra,JLD

function save_ansatz(path,ansatz)
    save(joinpath(@__DIR__,path),"qc",ansatz)
end

function get_hj(hlist)
    obs = reduce(kron,hlist)
    return obs|> SparseMatrixCSC
end

function get_data(data_path)
    data = readdlm(data_path)
    nqubit = size(data)[2]-1
    return data,nqubit
end

# 0.9945510539990559 I I I I I X
# -0.10000000000000005 I I Z Z I I

function get_obs(data,thresholds)
    # # data_path_i = joinpath(@__DIR__,"data/h_test.txt")
    # data = readdlm(data_path)
    
    nqubit = size(data)[2]-1
    m = size(data)[1]
    
    H_list = []
    for i=1:m
        if abs(data[i,1]) < thresholds
            continue
        else
            h =  []
            for j=2:nqubit+1
                if data[i,j] == "X"
                    push!(h,mat(X))
                elseif data[i,j] == "Y"
                    push!(h,mat(Y))
                elseif data[i,j] == "Z"
                    push!(h,mat(Z))
                elseif data[i,j] == "I"
                    push!(h,IMatrix(2))
                end
            end
            push!(H_list,(data[i,1],h))
        end
    end
    
    obs = zeros(ComplexF64,(1<<nqubit,1<<nqubit))|> SparseMatrixCSC

    for k=1:length(H_list) obs += H_list[k][1]*get_hj(H_list[k][2]) end

    return obs
end