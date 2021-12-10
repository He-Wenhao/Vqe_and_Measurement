using Yao,Plots,DataFrames,CSV,Tables

# function save_data(system_size::Int, project_path::String, type::String, data)
#     println("save data: $system_size qubit $type")
#     path = "./$project_path" * "/data/" * "$system_size" * "qubit:" * "$type" * ".csv"
#     CSV.write(path,Tables.table(data),writeheader=false)
# end

function save_data(system_size::Int, project_path::String, type::String, data...)
    # println("save data: $system_size qubit $type")
    df = DataFrame()
    l = length(data)
    for i=1:l
        if typeof(data[i]) <: Vector
            colname = colname = "D$i"
            df[!,colname] = data[i]
        else
            println(data[i],typeof(data[i]))
            (m,n) = size(data[i])
            for j=1:n
                colname = "D$i$j"
                df[!,colname] = data[i][:,j]
            end
        end
    end

    path = "$project_path" * "/data/" * "$system_size" * "qubit:" * "$type" * ".csv"
    # path = "./$project_path" * "/data/" * "$system_size" * "qubit:" * "$type" * ".csv"
    CSV.write(path,df)
end

function save_matrix(system_size::Int, project_path::String, type::String, data)
    path = "$project_path" * "/data/" * "$system_size" * "qubit:" * "$type" * ".csv"
    df = DataFrame(Matrix(data), :auto)
    CSV.write(path,df)
end

function savefigure(system_size::Int, project_path::String,type::String,indx,data...)
    # println("save figure: $system_size qubit $type")
    plt = plot(indx,data[1],lw=2)
    l = length(data)
    for n = 2:l
        plt = plot!(indx,data[n],shape=:circle,lw=2)
    end
    path = "$project_path" * "/figures/" * "$system_size" * "qubit:" * "$type" * ".png"
    savefig(plt,path)
end

function save_text(system_size::Int, project_path::String, type::String, data)
    df = DataFrame(data = data)
    path = "$project_path" * "data/" * "$system_size" * "qubit:" * "$type" * ".txt"
    CSV.write(path,df,append=true)
end


pair_ring(n::Int) = [i=>mod(i, n)+1 for i=1:n]
pairs = pair_ring(5)

function product(x::AbstractMatrix,y::AbstractMatrix)
    return y*x    
end

function den_state_zero()::AbstractMatrix
    st0 = statevec(zero_state(1))
    return st0*st0'
end

function den_state_one()::AbstractMatrix
    st1 = statevec(zero_state(1))
    st1[1] = 0
    st1[2] = 1
    return st1*st1'
end

function vec_state_zero(size::Int)
    s = zeros(ComplexF64,2^size)
    s[1] = 1
    return s
end

function vec_state_uniform(size::Int)
    N = 2^size
    s = ones(ComplexF64,N)
    return 1/sqrt(N) * s
end

function progressbar(current,total,type)
    width = 40
    percent = current / total
    p = percent * 100
    s = Int(round(width * percent))
    remaining = total - current
    bar = type * "[" * '='^s * ">" * ' '^(width - s) * "]" * "$p%" * "-remain:$remaining"
    println(bar)
end





