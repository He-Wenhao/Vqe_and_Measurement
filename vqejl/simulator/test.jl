include("qcircuit.jl")
include("qutils.jl")

#--------test create qcircuit with size = n---------
n = 4
c = Qcircuit(n,[])
for i=1:n
    add_gate(c,rotx(1,i))
end
add_gate(c,mcnot([n,1]))

#--------test get the matrix of qcircuit
m = get_mat_qc(c)
tmp = 1
for i=1:n
    gi = rotx(1,i).matrix
    tmp = kron(tmp,gi)
end
tmcnot = mcnot(n,[3,1])
m1 = tmp* expand_multi(n,tmcnot,false)
sum(m1 - m)

#-------test random initialize the param of qcircuit
rand_initialize_param!(c)

#-------test expand the 
t = rotx(π/5,2)
t = σx(1)
expand_block(3,t,false) * statevec(ArrayReg(bit"100"))
# expand_block(3,t) * statevec(ArrayReg(bit"100"))
# ss = mat(rx(π/2))
# # a = kron(mat(rx(π/2)),IMatrix(2^5))
# m = kron(a,a)

#------test gradient of each gate
n = 3
c = Qcircuit(n,[])
g0 = σx(2)
g1 = rotx(π/6,1)
g2 = mcnot(n,[3,2])
add_gate(c,g0)
add_gate(c,g1)
add_gate(c,g2)
show_circuit(c)
m = get_grad_state(c,vec_state_one(n))
sizeof(m)
mideal = kron(-1im/2*mat(X)*g1.matrix,kron(IMatrix(2),IMatrix(2)))*kron(IMatrix(2),kron(g0.matrix,IMatrix(2)))*expand_block(3,g2,false)
sum(m[2] - mideal)

#------test expval
include("qcircuit.jl")
include("hamiltonian.jl")
include("qutils.jl")

n = 8
c = Qcircuit(n,[])
k = 0
nlayer = round(n-1)
for l=1:nlayer
    for i=1:n
        add_gate(c,rotz(rand(),i))
        add_gate(c,roty(rand(),i))
        add_gate(c,rotz(rand(),i))
    end
    for j=1:n
        add_gate(c,mcnot(n,[j,j%n+1]))
    end
end
# show_circuit(c)
# get_mat_qc(c)
@time st = vec_state_uniform(n)
# # @timev evaluate(c,st)
@timev res = get_mat_qc(c)*st
typeof(res)

##---------test initialize
include("qcircuit.jl")
include("hamiltonian.jl")
xx = rotx(0,1)
yy = roty(0,1)
zz = rotz(0,1)

n = 10
c = Qcircuit(n,[])
Hmatrix = Hn(n)
add_gate(c,roth(rand(),n,Hmatrix))
for i=1:n
    add_gate(c,rotx(rand(),i))
end
length(c.gatelist[1].wires)
show_circuit(c)

n = 9
c = Qcircuit(n,[])
for k = 1:n-1
    for j=1:n
        add_gate(c,rotx(rand(),j))
        add_gate(c,roty(rand(),j))
        add_gate(c,rotz(rand(),j))
    end

    for j=1:n
        add_gate(c,mcnot(n,[j,j%n+1]))
    end
end
show_circuit(c)

st = vec_state_uniform(n)
m = get_mat_qc(c)
st1 = evaluate(c,st)
st2 = m * st
obs = IMatrix(1<<n) - st * st'
res = st1' * obs * st1
real(res)

#---------test gate
include("qcircuit.jl")
include("hamiltonian.jl")
include("gate.jl")
n=3
Hmatrix = Hn(n)
c = roth(π/6,n,Hmatrix)
size(c.matrix)

a = rand(3,2)
Q,R=qr(a)

#---------- test save save_circuit
include("qcircuit.jl")
include("hamiltonian.jl")
include("gate.jl")
using JLD
n = 7
c = Qcircuit(n,[])
for k = 1:n-1
    for j=1:n
        add_gate!(c,rotx(rand(),j))
        add_gate!(c,roty(rand(),j))
        add_gate!(c,rotz(rand(),j))
    end

    for j=1:n
        add_gate!(c,mcnot(n,[j,j%n+1]))
    end
end
show_circuit(c)

save(joinpath(@__DIR__,"circuit/circuit.jld"),"qc",c)
qc = load(joinpath(@__DIR__,"circuit/circuit.jld"))
