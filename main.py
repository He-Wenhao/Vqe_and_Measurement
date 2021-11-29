from Pauli_Expectation import *
from HamTN import *
from TN_VQE import *
import time
import qiskit.quantum_info as qi

if __name__ == '__main__':
    n = 10  # qubits
    g = 10  # parameter of Ising
    J = 0.1  # parameter of Ising
    l = 1  # layers
    n_para = UniTN_1d_para_cnt(n, l)  # nums of parameters
    paraList = [random.uniform(1, 10) for i in range(n_para)]

    start = time.time()
    tn_tmp = get_Ham_Pauli_SVD_v1(n, g, J, l, paraList, 0)

    for i in range(4 ** n):
        st = generate_pauli_str(i, n)
        if if_non_zero_Pauli(st, l):
            res = str(get_Ham_Pauli_SVD_v2(tn_tmp, st, 1)) + " " + st.replace('', ' ').strip()
            print(res)
    end = time.time()
    time2 = (end - start)

    print('n = ', n, "time = ", time2, "(new)")

    start = time.time()
    H = qi.DensityMatrix(get_Ham(n, g, J, l, paraList))  # get Hamiltonian matrix
    mid = time.time()
    delt1 = mid - start
    pauli_trace(H, n, l)
    end = time.time()
    delt2 = end - mid
    print(delt1, delt2)

    '''start = time.time()
    print(Ansatz_optimization(n))
    end = time.time()
    delt = end - start
    print(delt)

    print(spectrum_1dTransverse_Ising(n, g, J))
'''