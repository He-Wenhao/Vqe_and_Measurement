using Yao,LuxurySparse
include("const.jl")


export rotx,roty,rotz,σx,σy,σz,update_gate!

mutable struct Gate
    params
    wires
    matrix
    base
    grad
    name
    function Gate(params, wires, matrix, base, grad, name)
        return new(params, wires, matrix, base, grad, name)
    end
end

function mCNOT(size, wires, Mx)
    a = findmin(wires)[1]
    b = findmax(wires)[1]
    ctrl = wires[1] - a
    targ = wires[2] - a
    s = b - a + 1
    if ctrl < targ
        C1 = kron(P0,IMatrix(1<<(s-1)))
        nbetween = targ - ctrl - 1
        mbetween = IMatrix(1<<nbetween)
        C2 = kron(P1,kron(mbetween,Mx))
    else
        C1 = kron(IMatrix(1<<(s-1)),P0)
        nbetween = ctrl - targ - 1
        mbetween = IMatrix(1<<nbetween)
        C2 = kron(Mx,kron(mbetween,P1))
    end

    uc = C1+C2
    nbefore = a - 1
    nafter  = size - b
    U = kron(IMatrix(1<<nbefore),kron(uc,IMatrix(1<<nafter)))
    return U
end



function mGate(size,wires,gate) #---- control_U 
    @assert length(wires) > 1 "dimension not match(please check length of wires)"
    N = size
    ctrl =  wires[1]
    targ = wires[2]
    ctrl_list = [Matrix(I2) for i=1:N]
    targ_list = [Matrix(I2) for i=1:N]
    ctrl_list[ctrl] = P0
    targ_list[ctrl] = P1
    targ_list[targ] = gate
    U = reduce(kron,ctrl_list)+reduce(kron,targ_list)
    U |> SparseMatrixCSC
end

function gradGX(θ,name::String)
    if name == "rx"
        return -1/2*im*mat(X)*mat(Rx(θ))
    elseif name == "ry"
        return -1/2*im*mat(Y)*mat(Ry(θ))
    elseif name == "rz"
        return -1/2*im*mat(Z)*mat(Rz(θ))
    end
end

function gradGX(θ,matrix::AbstractMatrix)
    n = size(matrix)
    # println(-1/2*sin(θ/2)*IMatrix(n[1]) - 1/2*1im*cos(θ/2)*matrix)
    return -1/2*sin(θ/2)*IMatrix(n[1]) - 1/2*1im*cos(θ/2)*matrix 
end

function rotH(θ,wires,hm)
    @assert wires[1] == wires[2] "roth the wire[1] should equal to wire[2]"
    size = wires[1]
    return (cos(θ/2)*IMatrix(1<<size) - 1im*sin(θ/2)*hm) |> SparseMatrixCSC
end

function phaseCorrect(nqubit,θ)
    N = 1<<nqubit
    V = Matrix((1.0+0im)*I,N,N) |> SparseMatrixCSC
    ϕ = acot(1/sqrt(N)*tan(θ))
    V[1,1]=exp(-1im*ϕ)
    return V
end

p = phaseCorrect(2,π/6)

function spGate(θ,wires,hm)
    @assert wires[1] == wires[2] "roth the wire[1] should equal to wire[2]"
    nqubit = wires[1]
    V = phaseCorrect(nqubit,θ) 
    U = rotH(θ,wires,hm) 
    return V*U
end

function gradSP(θ,wires,hm)
    @assert wires[1] == wires[2] "roth the wire[1] should equal to wire[2]"
    nqubit = wires[1]
    N = 1<<nqubit

    Vg = zeros(ComplexF64,N,N) |> SparseMatrixCSC
    ϕ = acot(1/sqrt(N)*tan(θ))
    Vg[1,1]= -1im*exp(-1im*ϕ) * -1*sqrt(nqubit)*sec(θ)^2/(nqubit+tan(θ)^2)
    V = phaseCorrect(nqubit,θ)

    Ug = gradGX(θ,hm)|>SparseMatrixCSC
    U = rotH(θ,wires,hm)

    gradU = Vg*U + V*Ug
    return gradU
end

# a,b = gradSP(π/6,[2,2],Hn(2))

function update_gate!(g::Gate,param)
    g.params = param
    if g.name == "rx"
        g.matrix = mat(Rx(param))
        g.grad = gradGX(param,"rx")
    elseif g.name == "ry"
        g.matrix = mat(Ry(param))
        g.grad = gradGX(param,"ry")
    elseif g.name == "rz"
        g.matrix = mat(Rz(param))
        g.grad = gradGX(param,"rz")
    elseif g.name == "rh"
        g.matrix = rotH(param,g.wires,g.base)
        g.grad = gradGX(param,g.base)
    end
    # print("------up--",g)
    return g
end

function Hn(n::Int)
    M = 1
    mm = mat(H)
    for i = 1:n
        M = kron(M,mm)
    end
    return M
end

function zzMapping(size::Int,order_type::String,params,wires)
    l = []
    push!(l,mcnot(size,wires))
    targ = wires[2]
    theta = (π - params[1])*(π - params[2])
    if order_type == "reverse"
        uz = rotz(theta,targ)
        uz.matrix = uz.matrix'
        push!(l,uz)
    elseif order_type == "order"
        push!(l,rotz(theta,targ))
    end
    push!(l,mcnot(size,wires))
    return l
end

σx(wires) = Gate(nothing,wires,mat(X),nothing,nothing,"x")
σy(wires) = Gate(nothing,wires,mat(Y),nothing,nothing,"y")
σz(wires) = Gate(nothing,wires,mat(Z),nothing,nothing,"z")

hardmard(wires) = Gate(nothing,wires,mat(H),nothing,nothing,"h")

# rotx(θ,wires) = Gate(θ,wires,mat(Rx(θ)),mat(X),gradGX(θ,"rx"),"rx")
# roty(θ,wires) = Gate(θ,wires,mat(Ry(θ)),mat(Y),gradGX(θ,"ry"),"ry")
# rotz(θ,wires) = Gate(θ,wires,mat(Rz(θ)),mat(Z),gradGX(θ,"rz"),"rz")

rotx(θ,wires) = Gate(θ,wires,mat(Rx(θ)),mat(X),gradGX(θ,mat(X)),"rx")
roty(θ,wires) = Gate(θ,wires,mat(Ry(θ)),mat(Y),gradGX(θ,mat(Y)),"ry")
rotz(θ,wires) = Gate(θ,wires,mat(Rz(θ)),mat(Z),gradGX(θ,mat(Z)),"rz")

roth(θ,wires,Hmatrix) = Gate(θ,wires,rotH(θ,wires,Hmatrix),Hmatrix,gradGX(θ,Hmatrix),"rh")
#params, wires, matrix, base, grad, name
spg(θ,wires,G) = Gate(θ,wires,spGate(θ,wires,G),G,gradSP(θ,wires,G),"sp")

mcnot(size,wires) = Gate(nothing, wires, mCNOT(size,wires,mat(X)), nothing, nothing,"cnot")

function nqubitLocal(size,nlayers,params=nothing)
    l = []
    for layer_i=1:nlayers
        #--------- rot layer
        if typeof(params) != Nothing
            for j=1:size
                push!(l,rotx(params[j],j))
                push!(l,roty(params[j],j))
                push!(l,rotz(params[j],j))
            end
        else
            for j=1:size
                push!(l,rotx(rand(),j))
                push!(l,roty(rand(),j))
                push!(l,rotz(rand(),j))
            end
        end
        #--------- entanglement layer
        for j=1:size
            for k=j+1:size
                push!(l,mcnot(size,[j,k]))
            end
        end

    end
    #--------- rot layer
    if typeof(params) != Nothing
        for j=1:size
            push!(l,rotx(params[j],j))
            push!(l,roty(params[j],j))
            push!(l,rotz(params[j],j))
        end
    else
        for j=1:size
            push!(l,rotx(rand(),j))
            push!(l,roty(rand(),j))
            push!(l,rotz(rand(),j))
        end
    end

    return l
end

function featureMapping(size::Int,order_type::String,params)
    l = []
    
    if order_type == "reverse"
        #------- Uzz layer
        uzz_list = []
        for i=1:size
            for j=i+1:size
                p = (params[i],params[j])
                zz = zzMapping(size,"reverse",p,[i,j])
                for g in zz push!(uzz_list,g) end
            end
        end
        for g in reverse!(uzz_list) push!(l,g) end

        #------- Uz layer
        for i=1:size
            rz = rotz(params[i],i)
            rz.matrix = rz.matrix'
            push!(l,rz)
        end

        #------- Uh layer
        for i=1:size push!(l,hardmard(i)) end

    elseif order_type == "order"
        #------- Uh layer
        for i=1:size push!(l,hardmard(i)) end

        #------- Uz layer
        for i=1:size push!(l,rotz(params[i],i)) end

        #------- Uzz layer
        for i=1:size
            for j=i+1:size
                p = (params[i],params[j])
                zz = zzMapping(size,"order",p,[i,j])
                for g in zz push!(l,g) end
            end
        end
    end

    return l
end