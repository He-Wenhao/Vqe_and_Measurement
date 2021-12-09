from Pauli_Expectation import *
from HamTN_3_para import *
from TN_VQE import *
import time
import qiskit.quantum_info as qi
import test_txt
import numpy as np
import tensornetwork as tn
from julia.api import Julia

jl = Julia(compiled_modules=False)
from julia import Main as jl

jl.include("./vqejl/vqe/vqe.jl")


def test1main():
    n = 4  # qubits
    g = 10  # parameter of Ising
    J = 0.1  # parameter of Ising
    l = 1  # layers
    n_para = UniTN_1d_para_cnt(n, l, 3)  # nums of parameters
    paraList = [random.uniform(1, 10) for i in range(n_para)]

    start = time.time()
    tn_tmp, _ = get_Ham_Pauli_SVD_v1(n, g, J, l, paraList, derive_TN=None)

    Pauli_list = generate_non_zero_Pauli(n, l)
    # Pauli_list = [generate_pauli_str(i, n) for i in range(4**n)]
    with open("test.txt", "w") as f:
        for st in Pauli_list:
            if if_non_zero_Pauli(st, l):
                res = str(
                    get_Ham_Pauli_SVD_v2(tn_tmp, st, contract_flag=0, derive_TN=None).real / 2 ** n) + " " + st.replace(
                    '', ' ').strip()
                f.write(res + "\n")
    end = time.time()
    time2 = (end - start)
    print('n = ', n, "time = ", time2, "(new)")
    print(Ansatz_optimization(n, need_optimize=False))
    print(spectrum_1dTransverse_Ising(n, g, J))


def test2main():
    n = 4  # qubits
    g = 10  # parameter of Ising
    J = 0.1  # parameter of Ising
    l = 1  # layers
    n_para = UniTN_1d_para_cnt(n, l, 3)  # nums of parameters
    paraList = [random.uniform(1, 10) for i in range(n_para)]

    start = time.time()
    tn_tmp, all_TN = get_Ham_Pauli_SVD_v1(n, g, J, l, paraList, derive_TN=None)
    myMat = TN2Matrix((tn_tmp, all_TN[1], all_TN[2]))
    print(np.linalg.eigh(np.array(myMat))[0])


def test3main():
    n = 4  # qubits
    g = 10  # parameter of Ising
    J = 0.1  # parameter of Ising
    l = 1  # layers
    matrix_test = test_txt.test_txt(n)
    print(np.linalg.eigh(np.array(matrix_test))[0])
    print(spectrum_1dTransverse_Ising(n, g, J))


def optimize_energy(n, g, J, l, paraList):
    start = time.time()
    tn_tmp, _ = get_Ham_Pauli_SVD_v1(n, g, J, l, paraList, derive_TN=None)
    Pauli_list = generate_non_zero_Pauli(n, l)
    # Pauli_list = [generate_pauli_str(i, n) for i in range(4**n)]
    with open("test.txt", "w") as f:
        for st in Pauli_list:
            val = get_Ham_Pauli_SVD_v2(tn_tmp, st, contract_flag=0, derive_TN=None).real / 2 ** n
            if abs(val) > 0.00001:
                res = str(val) + " " + st.replace('', ' ').strip()
                f.write(res + "\n")
    end = time.time()
    time2 = (end - start)
    res = jl.optimize("../../test.txt")
    print('n = ', n, "energy time = ", time2)
    return res


def compute_derivative(n, g, J, l, paraList):
    result = []
    start = time.time()
    for i in range(len(paraList)):
        tn_tmp, _ = get_Ham_Pauli_SVD_v1(n, g, J, l, paraList, derive_TN=i)
        Pauli_list = generate_non_zero_Pauli(n, l)
        # Pauli_list = [generate_pauli_str(i, n) for i in range(4**n)]
        with open("test.txt", "w") as f:
            for st in Pauli_list:
                val = get_Ham_Pauli_SVD_v2(tn_tmp, st, contract_flag=0, derive_TN=i).real / 2 ** n
                if abs(val) > 0.00001:
                    res = str(val) + " " + st.replace('', ' ').strip()
                    f.write(res + "\n")
        end = time.time()
        result.append(jl.calc_energy("../../test.txt"))
    time2 = (end - start)
    # print('n = ', n, "der time = ", time2)
    return result


def updateTheta(n, g, J, l, paraList):
    h = 5e-4
    h_for_debug = 1e-6
    der = compute_derivative(n, g, J, l, paraList)
    der_fin_diff = []
    for i in range(len(paraList)):
        p2 = paraList.copy()
        p2[i] += h_for_debug
        res = (cal_energy(n, g, J, l, p2) - cal_energy(n, g, J, l, paraList))/h_for_debug
        der_fin_diff.append(res)
    for i in range(len(paraList)):
        paraList[i] -= h * der[i]
        print("der debug:",der[i],'(',der_fin_diff[i],')')


def cal_energy(n, g, J, l, paraList):
    tn_tmp, _ = get_Ham_Pauli_SVD_v1(n, g, J, l, paraList, derive_TN=None)
    Pauli_list = generate_non_zero_Pauli(n, l)
    # Pauli_list = [generate_pauli_str(i, n) for i in range(4**n)]
    with open("test.txt", "w") as f:
        for st in Pauli_list:
            val = get_Ham_Pauli_SVD_v2(tn_tmp, st, contract_flag=0, derive_TN=None).real / 2 ** n
            if abs(val) > 0.00001:
                res = str(val) + " " + st.replace('', ' ').strip()
                f.write(res + "\n")
    return jl.calc_energy("../../test.txt")


def optimize_theta(n, g, J, l, paraList, loop, epsi):
    for i in range(loop):
        pre_energy = cal_energy(n, g, J, l, paraList)
        updateTheta(n, g, J, l, paraList)
        energy = cal_energy(n, g, J, l, paraList)
        print(str(i) + " " + str(energy))
        if abs(energy - pre_energy) < epsi:
            return energy, i
    return energy, loop


def main_loop():
    n = 5  # qubits
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
    print(cal_energy(n, g, J, l, theta))
    pre_TN = 999
    pre_quantum = 999
    for i in range(20):
        print('------- loop', i, '--------')
        energy_TN = optimize_theta(n, g, J, l, theta, 50, 1e-5)
        energy_quantum = optimize_energy(n, g, J, l, theta)
        if abs(energy_TN[0] - pre_TN) + abs(energy_quantum[0] - pre_quantum) < 1e-5:
            print(energy_quantum[0], " ", i)
        else:
            print("TN: ", energy_TN)
            print('quantum :', energy_quantum)
            print('theta :', theta)
            pre_TN = energy_TN[0]
            pre_quantum = energy_quantum[0]


if __name__ == '__main__':
    main_loop()
