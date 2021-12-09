
using Yao,LuxurySparse,LinearAlgebra

export H1,H0,build_Hamiltonian,ideal_ground_val_vec,obs_H0,obs_H1

function obs_H1(size::Int, indx)
    N = 2^size
    @assert indx <= N "index out bound"
    @assert indx > 0 "index should > 0"
    M = zeros(N,N)
    M[indx,indx] = 1
    return IMatrix(N) - M
end 

function obs_H0(size::Int)
    N = 2^size
    M = 1/N * ones(N,N)
    return IMatrix(N) - M
end

function build_Hamiltonian(size::Int, s, H0, H1)
    N = 2^size
    Im = IMatrix(N)
    H = (1-s)*H0 + s*H1
    return H
end

function ideal_ground_val_vec(Hamiltonian,type)
    if type == "eigenvalue"
        eig_val = eigvals(Hamiltonian)
        return eig_val[1]
    else
        eig_vec = eigvecs(Hamiltonian)
        return eig_vec[1]
    end
end



