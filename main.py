from Pauli_Expectation import *
from HamTN_3_para import *
from TN_VQE import *
import time
import qiskit.quantum_info as qi
import test_txt
import numpy as np
import tensornetwork as tn


def test1main():
    n = 4  # qubits
    g = 10  # parameter of Ising
    J = 0.1  # parameter of Ising
    l = 1  # layers
    n_para = UniTN_1d_para_cnt(n, l,3)  # nums of parameters
    paraList = [random.uniform(1, 10) for i in range(n_para)]

    start = time.time()
    tn_tmp, _ = get_Ham_Pauli_SVD_v1(n, g, J, l, paraList, derive_TN=None)

    Pauli_list = generate_non_zero_Pauli(n, l)
    #Pauli_list = [generate_pauli_str(i, n) for i in range(4**n)]
    with open("test.txt", "w") as f:
        for st in Pauli_list:
            if if_non_zero_Pauli(st, l):
                res = str(get_Ham_Pauli_SVD_v2(tn_tmp, st, contract_flag = 0,derive_TN=None).real/2**n) + " " + st.replace('', ' ').strip()
                f.write(res+"\n")
    end = time.time()
    time2 = (end - start)
    print('n = ', n, "time = ", time2, "(new)")
    print(Ansatz_optimization(n,need_optimize = False ))
    print(spectrum_1dTransverse_Ising(n, g, J))

def test2main():
    n = 4  # qubits
    g = 10  # parameter of Ising
    J = 0.1  # parameter of Ising
    l = 1  # layers
    n_para = UniTN_1d_para_cnt(n, l,3)  # nums of parameters
    paraList = [random.uniform(1, 10) for i in range(n_para)]

    start = time.time()
    tn_tmp, all_TN = get_Ham_Pauli_SVD_v1(n, g, J, l, paraList, derive_TN=None)
    myMat = TN2Matrix((tn_tmp,all_TN[1],all_TN[2]))
    print(np.linalg.eigh(np.array(myMat))[0])

def test3main():
    n = 4  # qubits
    g = 10  # parameter of Ising
    J = 0.1  # parameter of Ising
    l = 1  # layers
    matrix_test = test_txt.test_txt(n)
    print(np.linalg.eigh(np.array(matrix_test))[0])
    print(spectrum_1dTransverse_Ising(n, g, J))



def compute_energy(n,g,J,l,paraList,need_optimize):
    start = time.time()
    tn_tmp, _ = get_Ham_Pauli_SVD_v1(n, g, J, l, paraList, derive_TN=None)
    Pauli_list = generate_non_zero_Pauli(n, l)
    #Pauli_list = [generate_pauli_str(i, n) for i in range(4**n)]
    with open("test.txt", "w") as f:
        for st in Pauli_list:
            if if_non_zero_Pauli(st, l):
                res = str(get_Ham_Pauli_SVD_v2(tn_tmp, st, contract_flag = 0,derive_TN=None).real/2**n) + " " + st.replace('', ' ').strip()
                f.write(res+"\n")
    end = time.time()
    time2 = (end - start)
    res = Ansatz_optimization(n,need_optimize = need_optimize)
    print('n = ', n, "energy time = ", time2)
    return res

def compute_derivative(n,g,J,l,paraList):
    result = []
    start = time.time()
    for i in range(len(paraList)):
        tn_tmp, _ = get_Ham_Pauli_SVD_v1(n, g, J, l, paraList, derive_TN=i)
        Pauli_list = generate_non_zero_Pauli(n, l)
        #Pauli_list = [generate_pauli_str(i, n) for i in range(4**n)]
        with open("test.txt", "w") as f:
            for st in Pauli_list:
                if if_non_zero_Pauli(st, l):
                    res = str(get_Ham_Pauli_SVD_v2(tn_tmp, st, contract_flag = 0,derive_TN=i).real/2**n) + " " + st.replace('', ' ').strip()
                    f.write(res+"\n")
        end = time.time()
        result.append(Ansatz_optimization(n,need_optimize = False))
    time2 = (end - start)
    print('n = ', n, "der time = ", time2)
    return result
    

def updateTheta(n,g,J,l,paraList):
    h = 0.01
    der = compute_derivative(n,g,J,l,paraList)
    for i in range(len(paraList)):
        paraList[i] -= h*der[i]

def main_loop():
    n = 6  # qubits
    g = 10  # parameter of Ising
    J = 0.1  # parameter of Ising
    l = 1  # layers
    ## for debug, accurate diagonalization
    print(spectrum_1dTransverse_Ising(n, g, J))
    # parameter for quantum
    phi = []
    # parameter for TN
    n_theta = UniTN_1d_para_cnt(n, l, 3)  # nums of parameters
    theta = [random.uniform(1, 10) for i in range(n_theta)]
    print('energy :',compute_energy(n,g,J,l,theta,need_optimize = True).real)
    for i in range(20):
        print('------- loop',i,'--------')
        updateTheta(n,g,J,l,theta)
        print('energy :',compute_energy(n,g,J,l,theta,need_optimize = True).real)
        print('theta :',theta)
    


if __name__ == '__main__':
    main_loop()
