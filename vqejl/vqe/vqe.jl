include("../simulator/qcircuit.jl")
include("../simulator/hamiltonian.jl")
include("../simulator/qutils.jl")
include("vqe_utils.jl")


function get_grad(qc::Qcircuit,obs,state)
    N = get_num_param(qc)
    A = zeros(N,N)
    C = zeros(N)
    st = get_qc_state(qc,state)

    grad_state_list = get_grad_state(qc,state) #gradU_k * state

    current = 0
    for j=1:N
        # progressbar(current,N,"get grad")

        gradstate_j = grad_state_list[j]
        C[j] = -2 * real(gradstate_j' * obs * st)

        for k=1:N
            gradstate_k = grad_state_list[k]
            A[j,k] = 2 * real(gradstate_j' * gradstate_k)
        end
        current += 1
    end

    v = eigvals(A)
    u = eigvecs(A)

    for i=1:length(v)
        if v[i] > 1e-6
            v[i] = 1/v[i]
        else
            v[i] = 0
        end
    end

    d = Diagonal(v)
    u_ = inv(u)
    A_ = u*d*u_

    grad = A_*C
    # println("------calc grad------")
    return grad
end

function update_param(qc::Qcircuit,obs,state)
    grad = get_grad(qc,obs,state)
    eta = 1e-1
    theta = get_qc_param(qc)
    new_theta = [t + eta * g for (t,g) in zip(theta,grad)]
    set_qc_params!(qc,new_theta)
end

function get_energy(qc::Qcircuit,obs,state)
    st = evaluate(qc,state)
    energy = st' * obs * st
    # println("------get_energy------")
    return real(energy)
end

function training(maxiter::Int,qc::Qcircuit,obs,state)
    energy_list = []
    indx_list = []

    ideal_energy = 0
    eng = 0
    last_eng = 0

    for i=1:maxiter+1
        if i == 1
            ideal_energy = ideal_ground_val_vec(obs|>Matrix,"eigenvalue")
            # println("---ideal energy: $ideal_energy ----")
        end

        last_eng = get_energy(qc,obs,state)

        update_param(qc,obs,state)
        # println("---------------update-param---------------------")
        eng = get_energy(qc,obs,state)
        # println("---------------energy: $last_eng $i---------------------")
        convergence = abs(eng - last_eng)
        push!(indx_list, i)
        push!(energy_list,eng)

        if convergence <= 1e-05
            break
        end

    end

    result_state = evaluate(qc,state)

    return result_state,indx_list,energy_list
end

function create_qcircuit(n::Int,nlayer::Int)
    qc = Qcircuit(n,[])
    for i=1:nlayer
        for j=1:n
            add_gate!(qc,rotx(π/5,j))
            add_gate!(qc,roty(π/5,j))
            add_gate!(qc,rotz(π/5,j))
        end

        for k=1:n
            add_gate!(qc,mcnot(n,[k,k%n+1]))
        end
    end
    show_circuit(qc)
    return qc
end

function main()
    print("Start:")
    #------- load hamiltonian file
    data,nqubit = get_data(joinpath(@__DIR__,"data/h2.txt"))

    #------- create variational ansatz
    c = Qcircuit(nqubit,[])
    nlayer = 2
    for l=1:nlayer
        for i=1:nqubit
            add_gate!(c,roty(rand(),i))
            add_gate!(c,rotz(rand(),i))
        end
        for j=1:nqubit-1
            add_gate!(c,mcnot(nqubit,[j,j+1]))
        end
    end
    show_circuit(c)

    #------- construct hamiltonian & calculate the ideal energy
    obs = get_obs(data,1e-5)
    ideal_energy = eigvals(obs|>Matrix)[1]

    state = vec_state_zero(nqubit)
    # obs1 = kron(IMatrix(1<<5),mat(X))*0.9945510539990559 + -0.10000000000000005*kron(IMatrix(1<<2),kron(kron(mat(Z),mat(Z)),IMatrix(1<<2)))
    result_state,indx_list,energy_list = training(500,c,obs,state)

    ideal_energy = eigvals(obs|>Matrix)

    savefigure(c.size,"vqe","expm energy ---",indx_list,energy_list)

    # save_data(c.size,"vqe","expm",indx_list,energy_list)
end

function optimize(path)
    # print("Start:")
    #------- load hamiltonian file
    # data,nqubit = get_data(joinpath(@__DIR__,"data/h2.txt"))
    data,nqubit = get_data(joinpath(@__DIR__,path))

    #------- create variational ansatz
    c = Qcircuit(nqubit,[])
    nlayer = 2
    for l=1:nlayer
        for i=1:nqubit
            add_gate!(c,roty(rand(),i))
            add_gate!(c,rotz(rand(),i))
        end
        for j=1:nqubit-1
            add_gate!(c,mcnot(nqubit,[j,j+1]))
        end
    end
    # show_circuit(c)

    #------- construct hamiltonian & calculate the ideal energy
    obs = get_obs(data,1e-5)
    ideal_energy = eigvals(obs|>Matrix)[1]

    state = vec_state_zero(nqubit)
    # obs1 = kron(IMatrix(1<<5),mat(X))*0.9945510539990559 + -0.10000000000000005*kron(IMatrix(1<<2),kron(kron(mat(Z),mat(Z)),IMatrix(1<<2)))
    result_state,indx_list,energy_list = training(500,c,obs,state)

    ideal_energy = eigvals(obs|>Matrix)

    savefigure(c.size,"vqejl/vqe","expm energy ---",indx_list,energy_list)

    # save_data(c.size,"vqe","expm",indx_list,energy_list)
    #---save ansatz
    save_ansatz(joinpath(@__DIR__,"circuit/circuit.jld"),c)
    #---last energy
    # println("length=======>",length(energy_list))
    save_data(nqubit,joinpath(@__DIR__,""),"energy",energy_list)

    return last(energy_list), length(energy_list)
end


function calc_energy(path, loop)
    data,nqubit = get_data(joinpath(@__DIR__,path))
    obs = get_obs(data,1e-5)
    state = vec_state_zero(nqubit)
    if loop != 0
        qc = load(joinpath(@__DIR__,"circuit/circuit.jld"))
        energy = get_energy(qc["qc"],obs,state)
    else
        energy = real(state'*obs*state)
    end
    return energy
end


