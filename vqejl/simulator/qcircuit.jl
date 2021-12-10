using LuxurySparse, LinearAlgebra, SparseArrays
export Qcircuit,
    set_gatelist,
    get_gatelist,
    add_gate,
    set_qc_params!,
    rand_initialize_param!,
    get_num_param,
    get_mat_qc,
    get_grad_gate

include("gate.jl")
include("qutils.jl")

mutable struct Qcircuit
    size
    gatelist::Vector
    function Qcircuit(size::Int,gatelist::Vector)
        return new(size,gatelist)
    end
end

function add_gate!(c::Qcircuit, gate::Gate)
    return c.gatelist = push!(c.gatelist, gate)
end

function add_block!(c::Qcircuit, block::Vector)
    for g in block c.gatelist = push!(c.gatelist, g) end
end

function reset_circuit!(c::Qcircuit)
    c.gatelist = []
end

function get_qc_param(qc::Qcircuit)
    param =[]
    for g in qc.gatelist
        if typeof(g.params) != Nothing
            push!(param,g.params)
        end
    end
    return param
end

function show_circuit(c::Qcircuit)
    l = length(c.gatelist)
    if l == 0
        println("circuit is clean")
    else
        for i=1:l
            println("---(",c.gatelist[i].wires,")-params:",c.gatelist[i].params,"--: ",c.gatelist[i].name)
        end
    end
end

function save_circuit(c::Qcircuit)
    l = length(c.gatelist)
    for i=1:l
        if length(c.gatelist[i].wires) > 1
            w = string(c.gatelist[i].wires[1]) * "," * string(c.gatelist[i].wires[1])
        else
            w = string(c.gatelist[i].wires)
        end
        data = "---(" * w * ")-params:" * string(c.gatelist[i].params) * "--: " * c.gatelist[i].name
        data = data * "\n-------------------------------------------------"
        save_text(c.size,"qkernel","circuitinfo",data)
    end
    data = "\n============================================================"
    save_text(c.size,"qkernel","circuitinfo",data)
    println("save circuit done")

    params = [g.params for g in c.gatelist if typeof(g.params) != Nothing]
    save_data(c.size,"qkernel","circuit_params",[i for i=1:length(params)],params)
    println("save parameters done")
end

function set_qc_params!(c::Qcircuit,param)
    l = []
    for g in c.gatelist typeof(g.params) != Nothing ? push!(l,1) : push!(l,0) end
    N = sum(l)
    @assert length(param)==N "check the number of vqc parameters"
    l = 1
    len = length(c.gatelist)
    for k=1:len
        if typeof(c.gatelist[k].params) != Nothing
            # println("----set------",c.gatelist[k][2])
            update_gate!(c.gatelist[k],param[l])
            l+=1
        else
            continue
        end
    end
    # print("******************")
end

function set_gatelist(c::Qcircuit,g)
    c.gatelist = g
end

function get_gatelist(c::Qcircuit)
    return c.gatelist
end

function get_num_param(c::Qcircuit)
    l = []
    for g in c.gatelist typeof(g.params) != Nothing ? push!(l,1) : push!(l,0) end
    return sum(l)
end

function expand_m(a::Int,b::Int,na::Int,nb::Int,size::Int,gmatrix)
    a = min(a,b)
    b = max(a,b)
    gsize = b - a + 1
    if gsize == size
        return gmatrix
    else
        U = kron(IMatrix(1<<a), kron(gmatrix,IMatrix(1<<(size - b - 1))))
        return U
    end
end

# function expand_m(a::Int,b::Int,na::Int,nb::Int,size::Int,gmatrix)
#     if a == 1 && b == size
#         m_block = gmatrix
#     elseif a == 1
#         m_block = kron(gmatrix,IMatrix(2^nb))
#     elseif b == size
#         m_block = kron(IMatrix(2^na),gmatrix)
#     else
#         m_block = kron(IMatrix(2^na), kron(gmatrix,IMatrix(2^nb)))
#     end
#     return m_block
# end

function expand_multi(size::Int,gate::Gate,is_grad::Bool)
    a = findmin(gate.wires)[1]
    b = findmax(gate.wires)[1]
    na = a - 1
    nb = size - b
    if is_grad == false
        if size(gate.matrix) == (1<<size)
            gg = gate.matrix |> SparseMatrixCSC
            return gg
        else
            gg = gate.matrix |> SparseMatrixCSC
            U = kron(IMatrix(1<<na), kron(gg,IMatrix(1<<(nb))))
            return U
        end
    else
        if size(gate.grad) == (1<<size)
            gg = gate.grad |> SparseMatrixCSC
            return gg
        else
            gg = gate.grad |> SparseMatrixCSC
            U = kron(IMatrix(1<<na), kron(gg,IMatrix(1<<(nb))))
            return U
        end
    end
end

# function expan_1qubit_gates(gate_mat,size,wires):
#     U = 1
#     gate = gate_mat
#     for i in range(size):
#         if i == wires:
#             U = np.kron(U,gate)
#         else:
#             U = np.kron(U,np.eye(2))
#     return U
# end

function expand_s(size::Int, wires::Int, gmatrix)
    U = 1
    for i=1:size
        if i == wires
            U = kron(U,gmatrix)
        else
            U = kron(U,IMatrix(2))
        end
    end
    return U
end
#     if wires == 1
#         return kron(gmatrix , IMatrix(2^(size-1)))
#     elseif wires == size
#         return kron(IMatrix(2^(size-1)), gmatrix)
#     else
#         return kron(IMatrix(2^(wires-1)), kron(gmatrix , IMatrix(2^(size - wires))))
#     end
# end

# def expan_1qubit_gate(gate_mat,size,wires):
#     if wires == size - 1:
#         Im = np.eye(2**(size - 1))
#         U = np.kron(Im,gate_mat)
#     elif wires == 0:
#         Im = np.eye(2**(size - 1))
#         U = np.kron(gate_mat,Im)
#     else:
#         nbefore = wires
#         nafter = size - wires - 1
#         Ib = np.eye(2**nbefore)
#         Ia = np.eye(2**nafter)
#         U = np.kron(Ib,np.kron(gate_mat,Ia))
#     return U


function expand_single(size::Int,gate::Gate,is_grad::Bool)
    # println("----expsingle-----",gate.name,is_grad)
    U = 1
    for i=1:size
        if i == gate.wires
            if is_grad == false 
                U = kron(U,gate.matrix)
            else
                U = kron(U,gate.grad)
            end
        else
            U = kron(U,IMatrix(2))
        end
    end
    return U
end

function get_qc_state(c::Qcircuit,state)
    for g in c.gatelist
        if length(g.wires) > 1
            gg = g.matrix |> SparseMatrixCSC
            state = gg * state
        else
            if size(g.matrix)[1] == (1<<c.size)
                gg = g.matrix |> SparseMatrixCSC
                state = gg * state
            else
                state = expand_single(c.size,g,false) * state
            end
        end
    end
    return state
end

function get_mat_qc(c::Qcircuit)
    mm = IMatrix{1<<c.size}()
    for g in c.gatelist
        if length(g.wires) > 1
            if size(g.matrix)[1] == (1<<c.size)
                gg = g.matrix |> SparseMatrixCSC
                mm = gg * mm
            else
                mm = expand_multi(c.size,g,false) * mm
            end
        else
            mm = expand_single(c.size,g,false) * mm
        end
    end
    return mm
end

function evaluate(c::Qcircuit,state)
    for g in c.gatelist
        if length(g.wires) > 1
            if size(g.matrix)[1] == (1<<c.size)
                gg = g.matrix |> SparseMatrixCSC
                state = gg * state

                # w = g.wires
                # println("=====evaluate==$w=={" * g.name * "===             ",sum(state))
                # mat = expand_multi(c.size,g,false) * mat
            else
                w = g.wires
                println("=====evaluate==$w=={" * g.name * "===             ",sum(state))
                state = expand_multi(c.size, g, false) * state

                # w = g.wires
                # println("=====evaluate==$w=={" * g.name * "===             ",sum(state))
            end
        else
            state = expand_single(c.size, g, false) * state
            
            # w = g.wires
            # println("=====evaluate==$w=={" * g.name * "===             ",sum(state))
        end
    end
    return state
end

function expand_gate(system_size::Int, gate::Gate, is_grad::Bool)
    n = length(gate.wires)
    if n > 1
        if size(gate.matrix)[1] == (1<<system_size)
            if is_grad == false
                return gate.matrix
            else
                return gate.grad
            end            
        else
            return expand_multi(system_size,gate,is_grad)
        end
    else
        return expand_single(system_size,gate,is_grad)
    end
end

function rand_initialize_param!(c::Qcircuit)
    N = get_num_param(c)
    params = rand(Float64, N)                               
    show_circuit(c)
    set_qc_params!(c,params)
end

function get_grad_state(c::Qcircuit,state)
    grad_s = []
    N = length(c.gatelist)
    # current = 0
    for i=1:N
        st = deepcopy(state)
        # progressbar(current,N,"get grad matrix")
        if typeof(c.gatelist[i].params) != Nothing
            for j = 1:N
                if typeof(c.gatelist[j].params) != Nothing
                    if i == j 
                        st = expand_gate(c.size,c.gatelist[j],true) * st # get the grad of gate_i
                    else
                        st = expand_gate(c.size,c.gatelist[j],false) * st # get the matrix of gate_i
                    end
                else
                    # print("****ggg-----",c.gatelist[j].params)
                    st = expand_gate(c.size,c.gatelist[j],false) * st
                end
            end
            push!(grad_s, st)
        else
            continue
        end
        # current += 1
    end
    return grad_s
end

function get_grad_gate(c::Qcircuit)
    grad_g = []
    N = length(c.gatelist)
    # current = 0
    for i=1:N
        # progressbar(current,N,"get grad matrix")
        if typeof(c.gatelist[i].params) != Nothing
            mat_block = 1
            for j=1:N
                if typeof(c.gatelist[j].params) != Nothing
                    if i == j
                        mat_block = expand_gate(c.size,c.gatelist[j],true) * mat_block
                    else
                        mat_block = expand_gate(c.size,c.gatelist[j],false) * mat_block
                    end
                else
                    # print("****ggg-----",c.gatelist[j].params)
                    mat_block = expand_gate(c.size,c.gatelist[j],false) * mat_block
                end
            end
            push!(grad_g, mat_block)
        else
            continue
        end
        # current += 1
    end
    return grad_g
end

function get_energy_(c::Qcircuit, state, obs)
    U = get_mat_qc(c)
    return state'* U' * obs * U * state
end

# n = 4
# c = Qcircuit(n,[])
# for i=1:n
#     add_gate(c,rotz(rand(),i))
#     add_gate(c,roty(rand(),i))
#     add_gate(c,rotz(rand(),i))
# end
# for j=1:n
#     add_gate(c,mcnot(n,[j,j%n+1]))
# end
# get_mat_qc(c)

# show_circuit(c)
# t = mcnot(n,[3,4])
# t.matrix
# expand_multi(4,mcnot([3,4]),false)