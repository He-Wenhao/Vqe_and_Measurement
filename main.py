from Pauli_Expectation import *
from HamTN import *
from TN_VQE import *
import time
import qiskit.quantum_info as qi

if __name__ == '__main__':
    n = 5  # qubits
    g = 10  # parameter of Ising
    J = 0.1  # parameter of Ising
    l = 1  # layers
    n_para = UniTN_1d_para_cnt(n, l)  # nums of parameters
    paraList = [random.uniform(1, 10) for i in range(n_para)]
    start = time.time()
    H = qi.DensityMatrix(get_Ham(n, g, J, l, paraList))  # get Hamiltonian matrix
    pauli_trace(H, n)
    end = time.time()
    delt = end - start
    print(delt)

    '''start = time.time()
    print(Ansatz_optimization(n))
    end = time.time()
    delt = end - start
    print(delt)

    print(spectrum_1dTransverse_Ising(n, g, J))
'''