import tensornetwork as tn
import numpy as np
from scipy.linalg import expm
from math import sqrt
import random
from typing import List, Tuple
import itertools
import scipy.integrate as integrate
from scipy.optimize import minimize

##################################
## for debug
##################################
def debugprint(*para):
    if 1:
        print(*para)
def debug_22_22_IsUnitary(tens_p):
    if 1:
        tens = tens_p.copy()
        tens_conj = tens.copy(conjugate = True)
        tens[2]^tens_conj[2]
        tens[3]^tens_conj[3]
        nw = tn.contractors.auto([tens,tens_conj],output_edge_order=[tens[0],tens[1],tens_conj[0],tens_conj[1]])
        res = 0
        for i,j,k,l in [(i,j,k,l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]:
            if i == k and j == l:
                res += abs(nw.tensor[i,j,k,l] - 1)**2
            else:
                res += abs(nw.tensor[i,j,k,l])**2
        print("ten22_22 is unitary: ",sqrt(res)<1e-10)

def debug_UTN_IsUnitary(tens_p):
    if 1:
        tens = tens_p.copy()
        ind_cnt = len(tens.get_all_edges())
        tens_conj = tens.copy(conjugate = True)
        for i in range(ind_cnt//2):
            tens[ind_cnt//2+i] ^ tens_conj[ind_cnt//2+i]
        nw = tn.contractors.auto([tens,tens_conj],output_edge_order=tens[0:ind_cnt//2]+tens_conj[0:ind_cnt//2])
        ## now we want to check if nw is a unitary tensor
        s = list(itertools.product(range(2), repeat=ind_cnt//2))
        res = 0
        for i in s:
            for j in s:
                if i == j:
                    res += abs(nw.tensor[i+j] - 1)**2
                else:
                    res += abs(nw.tensor[i+j])**2
        print("||TN*TN^dagger - Identity||_2 =",sqrt(res),"; unitary: ",sqrt(res)<1e-10)

## get spectrum of 1-d transverse Ising model (boundary condition: finite chain)
def spectrum_1dTransverse_Ising(n:int,g:float,J:float) -> np.array: ####################
    inds = list(itertools.product(range(2), repeat=n))
    dim = len(inds)
    #res = dim*[dim*[0.]]
    res = [[0. for i in range(dim)] for j in range(dim)]
    ## first, we consider the 2-sites term
    for i in range(dim):
        ind = inds[i]
        for j in range(n-1):
            if ind[j] == ind[j+1]:
                res[i][i] -= J
            else:
                res[i][i] += J
    ## then we consider the 1-site term
    for i in range(dim):
        for j in range(dim):
            ai = inds[i]
            aj = inds[j]
            if sum([abs(ai[cnt]-aj[cnt]) for cnt in range(n)]) == 1:
                res[i][j] -= J*g
    #debugprint(res)
    #debugprint(np.linalg.eigh(np.array(res))[0])
    return np.linalg.eigh(np.array(res))[0]

## analytical solution of E0 in i-d transverse Ising model (boundary condition: periodic)
## https://zhuanlan.zhihu.com/p/421058781
def analytic_E0(n:int,g:float,J:float) -> float:
    h = g*J
    return -(n/2./np.pi)*integrate.quad(lambda x: sqrt(J**2+h**2+2*J*h*np.cos(x)), -np.pi, np.pi)[0]

## n qubits identity tensor
def Iden(n:int) -> Tuple[List[tn.Node],List[tn.Edge],List[tn.Edge]]:
    all_Nodes = [] ## to track all Nodes
    edge_i = n*[0]
    edge_j = n*[0]
    for i in range(n):
        iden = tn.Node(np.array([[1.+0.j,0.+0.j],[0.+0.j,1.+0.j]]))
        all_Nodes.append(iden)
        edge_i[i] = iden[0]
        edge_j[i] = iden[1]
    return (all_Nodes,edge_i,edge_j)

##################################


##  parameters(a list of 15 real numbers) -> unitary tensor U_{i1,i2;j1,j2}
def UniTen22_22(para : List[float],conjugate = False) -> tn.Node:
    #debug
    if len(para) != 15:
        raise TypeError("len(para) error")
    ## first, we construct a 4*4 unitary matrix by e^{S} where S is a 4*4 anti symmetry
    ##represent of H
    lamb = [
        np.array(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, -1.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        (1/sqrt(3))*np.array(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, -2.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        (1/sqrt(6))*np.array(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, -3.+0.j]]),
        np.array(
            [[0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
             [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j],
             [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
             [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j],
             [0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
             [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]]),
        np.array(
            [[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
             [0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
             [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j]])]
    H = sum([para[i]*lamb[i] for i in range(15)])
    U44 = expm(1.j * H)
    if conjugate == True:
        U44 = U44.conjugate()
    #debugprint(np.dot(np.asmatrix(U),np.asmatrix(U).getH()))
    #debugprint(U)
    ## construct Tensor using U
    U2222 = np.array(2*[2*[2*[2*[0.+0.j]]]])
    for i,j,k,l in [(i,j,k,l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]:
        U2222[i,j,k,l] = U44[2*i+j,2*k+l]
    tens = tn.Node(U2222)
    return tens

## use finite difference method
def DerUniTen22_22_Naive(para:list,vari:int) -> tn.Node:    #######################
    mypara = para.copy()
    step = 0.01
    U22_22x = UniTen22_22(mypara).tensor
    mypara[vari] += step
    U22_22x_delta = UniTen22_22(para).tensor
    difU22_22 = (U22_22x_delta - U22_22x)/step
    return tn.Node(difU22_22)

def UniTN_1d_para_cnt(n,l):
    if n%2 == 1:
        n_para = n//2*l*15## number of parameters
    else:
        if l%2 == 0:
            n_para = (n//2 + n//2 -1)*(l//2)*15
        else:
            n_para = (n//2 + n//2 -1)*(l//2)*15 + n//2 *15
    return n_para
## a Unitary, for l layers and n qubits -> unitary network U_{i1,...,in;j1,...jn}
def UniTN_1d(n: int,l: int,para : List[float], conjugate = False) -> Tuple[List[tn.Node],List[tn.Edge],List[tn.Edge]]:
    mypara = list(para).copy()
    all_Nodes = [] ## to track all Nodes
    ## indices of U_{i1,...,in;j1,...jn}
    edge_i = n*[0]
    edge_j = n*[0]
    n_para = UniTN_1d_para_cnt(n,l)

    if len(mypara) != n_para:
        print(n_para)
        raise TypeError("len(para) error")

    if n%2 == 1:
        ## generate the first layer
        idTen2_2 = tn.Node(np.eye(2))
        all_Nodes.append(idTen2_2)
        edge_i[0] = idTen2_2[0]
        edge_j[0] = idTen2_2[1]
        for cnt in range(n//2):
            paralist = [mypara.pop(0) for i in range(15)]
            uniTen22_22 = UniTen22_22(paralist,conjugate) 
            all_Nodes.append(uniTen22_22)
            edge_i[2*cnt+1] = uniTen22_22[0]
            edge_i[2*cnt+2] = uniTen22_22[1]
            edge_j[2*cnt+1] = uniTen22_22[2]
            edge_j[2*cnt+2] = uniTen22_22[3]
        ## generate the li_th layer
        for li in range(l-1):
            if li%2 == 1:
                for cnt in range(n//2):
                    paralist = [mypara.pop(0) for i in range(15)]
                    uniTen22_22 = UniTen22_22(paralist,conjugate)
                    all_Nodes.append(uniTen22_22)
                    edge_i[2*cnt+1]^uniTen22_22[2]
                    edge_i[2*cnt+2]^uniTen22_22[3]
                    edge_i[2*cnt+1] = uniTen22_22[0]
                    edge_i[2*cnt+2] = uniTen22_22[1]
            if li%2 == 0:
                for cnt in range(n//2):
                    paralist = [mypara.pop(0) for i in range(15)]
                    uniTen22_22 = UniTen22_22(paralist,conjugate)
                    all_Nodes.append(uniTen22_22)
                    edge_i[2*cnt+0]^uniTen22_22[2]
                    edge_i[2*cnt+1]^uniTen22_22[3]
                    edge_i[2*cnt+0] = uniTen22_22[0]
                    edge_i[2*cnt+1] = uniTen22_22[1]

    
    if n%2 == 0:
        ## generate the first layer
        for cnt in range(n//2):
            paralist = [mypara.pop(0) for i in range(15)]
            uniTen22_22 = UniTen22_22(paralist,conjugate) 
            all_Nodes.append(uniTen22_22)
            edge_i[2*cnt+0] = uniTen22_22[0]
            edge_i[2*cnt+1] = uniTen22_22[1]
            edge_j[2*cnt+0] = uniTen22_22[2]
            edge_j[2*cnt+1] = uniTen22_22[3]
        ## generate the li_th layer
        for li in range(l-1):
            if li%2 == 1:
                for cnt in range(n//2):
                    paralist = [mypara.pop(0) for i in range(15)]
                    uniTen22_22 = UniTen22_22(paralist,conjugate)
                    all_Nodes.append(uniTen22_22)
                    edge_i[2*cnt+0]^uniTen22_22[2]
                    edge_i[2*cnt+1]^uniTen22_22[3]
                    edge_i[2*cnt+0] = uniTen22_22[0]
                    edge_i[2*cnt+1] = uniTen22_22[1]
            if li%2 == 0:
                for cnt in range(n//2-1):
                    paralist = [mypara.pop(0) for i in range(15)]
                    uniTen22_22 = UniTen22_22(paralist,conjugate)
                    all_Nodes.append(uniTen22_22)
                    edge_i[2*cnt+1]^uniTen22_22[2]
                    edge_i[2*cnt+2]^uniTen22_22[3]
                    edge_i[2*cnt+1] = uniTen22_22[0]
                    edge_i[2*cnt+2] = uniTen22_22[1]

    # for debug       
    #resTN = tn.contractors.auto(all_Nodes,output_edge_order=edge_i+edge_j)
    #debug_UTN_IsUnitary(resTN)
    return (all_Nodes,edge_i,edge_j)

## use finite difference method
def DerUniTN_Naive(n: int,l: int,para:list,vari) -> tn.Node:    #######################
    mypara = para.copy()
    step = 0.01
    UniTN_x = UniTN_1d(n,l,para )
    mypara[vari] += step
    UniTN_x_delta = UniTen22_22(para).tensor
    difUniTN_x = (UniTN_x_delta - UniTN_x)/step
    return tn.Node(difUniTN_x)

## use chain rule
def DerUniTN_chain(n: int,l: int,para:list,vari) -> tn.Node:    #######################
    mypara = para.copy()
    step = 0.01
    UniTN_x = UniTN_1d(n,l,para )
    mypara[vari] += step
    UniTN_x_delta = UniTen22_22(para).tensor
    difUniTN_x = (UniTN_x_delta - UniTN_x)/step
    return tn.Node(difUniTN_x)


## H = -J \sum_i (Z_iZ_{i+1} + gX_i)

## cast 1-d transverse Ising model into MPO (boundary condition: finite chain)
## https://www.zhihu.com/search?type=content&q=MPO%20hamiltonian
def Ham2TN_1dTransverse_Ising(n:int,g:float,J:float) -> Tuple[List[tn.Node],List[tn.Edge],List[tn.Edge]]: ####################
    h = g*J
    sqJ = sqrt(J)
    if n < 2:
        raise TypeError("qubit number < 2")
    all_Nodes = []
    edge_i = [0 for i in range(n)]
    edge_j = [0 for i in range(n)]
    ## Pauli Matrices
    PX = np.array([[0.,1.],[1.,0.]])
    PZ = np.array([[1.,0.],[0.,-1.]])
    Id = np.array([[1.,0.],[0.,1.]])
    Zero = np.array([[0.,0.],[0.,0.]])
    ## matrix of tensor
    W1 = np.array([Id,-sqJ*PZ,-h*PX])
    Wn = np.array([-h*PX,sqJ*PZ,Id])
    Wi = np.array([
        [Id,-sqJ*PZ,-h*PX],
        [Zero,Zero,sqJ*PZ],
        [Zero,Zero,Id]
    ])
    ## construct network
    tnW1 = tn.Node(W1)
    all_Nodes.append(tnW1)
    edge_i[0] = tnW1[1]
    edge_j[0] = tnW1[2]
    contract_edge = tnW1[0]
    for i in range(n-2):
        tnWi = tn.Node(Wi)
        all_Nodes.append(tnWi)
        edge_i[i+1] = tnWi[2]
        edge_j[i+1] = tnWi[3]
        contract_edge ^ tnWi[0]
        contract_edge = tnWi[1]
    tnWn = tn.Node(Wn)
    all_Nodes.append(tnWn)
    edge_i[n-1] = tnWn[1]
    edge_j[n-1] = tnWn[2]
    contract_edge ^ tnWn[0]
    ## for debug
    #resTN = tn.contractors.auto(all_Nodes,output_edge_order=edge_i+edge_j)
    #print(resTN.tensor)
    return (all_Nodes,edge_i,edge_j)

## convert TN into matrix
def TN2Matrix(paratn: Tuple[List[tn.Node],List[tn.Edge],List[tn.Edge]]):
    # do contraction
    all_Nodes, edge_i, edge_j = paratn
    myTN = tn.contractors.auto(all_Nodes,output_edge_order=edge_i+edge_j)
    n = len(edge_i)
    res = np.array([[0.+0.j for i in range(2**n)] for j in range(2**n)])
    s = list(itertools.product(range(2), repeat=n))
    for i in range(2**n):
        for j in range(2**n):
            ij = s[i] + s[j]
            res[i][j] = myTN.tensor[ij]
    return res

## get Hamiltonian elemets
## n,g,J : config of Ising model
## l: config of Unitary Tensornetwork
## para: \phi parameters of Unitary tensor network
def get_Ham(n:int,g:float,J:float,l: int, para:list) -> np.array:
    mypara = para.copy()
    if len(mypara) != UniTN_1d_para_cnt(n,l):
        print(mypara)
        raise TypeError("len(para) error")
    # construct U H U^*
    HamTN = Ham2TN_1dTransverse_Ising(n,g,J) 
    #HamTN = Iden(n)
    UniTN = UniTN_1d(n,l,mypara)
    UniTN_conj = UniTN_1d(n,l,mypara,conjugate=True)
    # do U H U^\dagger
    for i in range(n):
        UniTN[2][i] ^ HamTN[1][i] 
        HamTN[2][i] ^ UniTN_conj[2][i]
    all_TN = (HamTN[0]+UniTN[0]+UniTN_conj[0],UniTN[1],UniTN_conj[1])
    # do contraction and cast into a matrix
    matrix_tens = TN2Matrix(all_TN) 
    return matrix_tens

#get DerHam for vari,using finite difference
def _get_DerHam_vari_Naive(n:int,g:float,J:float,l: int, para:list,vari: int) -> np.array:
    step = 0.001

    mypara = para.copy()
    if len(mypara) != UniTN_1d_para_cnt(n,l):
        print(mypara)
        raise TypeError("len(para) error")
    # construct U H U^*
    HamTN = Ham2TN_1dTransverse_Ising(n,g,J) 
    #HamTN = Iden(n)
    UniTN = UniTN_1d(n,l,mypara)
    UniTN_conj = UniTN_1d(n,l,mypara,conjugate=True)
    # do U H U^\dagger
    for i in range(n):
        UniTN[2][i] ^ HamTN[1][i] 
        HamTN[2][i] ^ UniTN_conj[2][i]
    all_TN = (HamTN[0]+UniTN[0]+UniTN_conj[0],UniTN[1],UniTN_conj[1])
    # do contraction and cast into a matrix
    matrix_tens1 = TN2Matrix(all_TN) 

    # do again with para + step

    # construct U H U^*
    mypara[vari] += step
    HamTN = Ham2TN_1dTransverse_Ising(n,g,J) 
    #HamTN = Iden(n)
    UniTN = UniTN_1d(n,l,mypara)
    UniTN_conj = UniTN_1d(n,l,mypara,conjugate=True)
    # do U H U^\dagger
    for i in range(n):
        UniTN[2][i] ^ HamTN[1][i] 
        HamTN[2][i] ^ UniTN_conj[2][i]
    all_TN = (HamTN[0]+UniTN[0]+UniTN_conj[0],UniTN[1],UniTN_conj[1])
    # do contraction and cast into a matrix
    matrix_tens2 = TN2Matrix(all_TN) 
    der = (matrix_tens2 - matrix_tens1) / step
    return der

## dH_jk/d\phi_i indices:ijk
def get_DerHam_Naive(n:int,g:float,J:float,l: int, para:list) -> List[np.array]:
    n_para = UniTN_1d_para_cnt(n,l)
    res = []
    for vari in range(n_para):
        res.append(_get_DerHam_vari_Naive(n,g,J,l, para,vari))
    return np.array(res)


# test unitary preserves eigenvalues
def test1():
    #UniTen22_22([random.uniform(1,10) for i in range(15)])
    #UniTN_1d(6,3,[random.uniform(1,10) for i in range(15*8)])
    #spectrum_1dTransverse_Ising(5,10,0.1)
    n = 5   # qubits
    g = 10  # parameter of Ising
    J = 0.1 # parameter of Ising
    l = 2   # layers
    n_para = UniTN_1d_para_cnt(n,l) # nums of parameters
    paraList = [random.uniform(1, 10) for i in range(n_para)]    # all parameters
    matrix_tens = get_Ham(n,g,J,l,paraList)
    print(np.linalg.eigh(np.array(matrix_tens))[0])
    print(spectrum_1dTransverse_Ising(n,g,J))

# test Derivative valid
# try to find the minimum energy, given a initial state
def test2():
    #UniTen22_22([random.uniform(1,10) for i in range(15)])
    #UniTN_1d(6,3,[random.uniform(1,10) for i in range(15*8)])
    #spectrum_1dTransverse_Ising(5,10,0.1)
    n = 4   # qubits
    g = 1  # parameter of Ising
    J = 1 # parameter of Ising
    l = 1   # layers
    n_para = UniTN_1d_para_cnt(n,l) # nums of parameters
    ini_para = [i**2 for i in range(n_para)]    # all parameters
    ini_state = np.array([i**3 for i in range(2**n)])
    # method1: use library in scipy
    def energy(mylist: List[float]) -> float:
        ham = get_Ham(n,g,J,l,mylist)
        state2 = np.dot(ham,ini_state)
        energy1 = np.dot(ini_state,state2)
        return energy1.real/np.dot(ini_state,ini_state)
    

    # method2: use derH
    def Derenergy(mylist: List[float]):
        DerHam = get_DerHam_Naive(n, g, J, l, mylist)
        result = []
        for der in DerHam:
            state2 = np.dot(der,ini_state)
            energy1 = np.dot(ini_state,state2)
            result.append(energy1.real/np.dot(ini_state,ini_state))
        return tuple(result)

    res2 = minimize(energy,ini_para,method='Newton-CG', tol=1e-3,jac = Derenergy)
    res1 = minimize(energy,ini_para,method='Nelder-Mead', tol=1e-3)
    print(energy(res1.x))
    print(energy(res2.x))
    print(spectrum_1dTransverse_Ising(n,g,J))

if __name__ == '__main__':
    #test1()
    #test2()

    n = 5   # qubits
    g = 10  # parameter of Ising
    J = 0.1 # parameter of Ising
    l = 2   # layers
    n_para = UniTN_1d_para_cnt(n,l) # nums of parameters
    get_Ham(n,g,J,l,n_para)     # get Hamiltonian matrix
    get_DerHam_Naive(n,g,J,l,n_para)    # get d H / d \phi
