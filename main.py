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
    n = 3  # qubits
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


def compute_difference(n, g, J, l, paraList, ifapply):
    result = []
    for i in range(len(paraList)):
        pre_energy = cal_energy(n, g, J, l, paraList, ifapply)
        paraList[i] += 1e-7
        energy = cal_energy(n, g, J, l, paraList, ifapply)
        diff = (energy-pre_energy)/1e-7
        result.append(diff)
        paraList[i] -= 1e-7
    return result



def compute_derivative(n, g, J, l, paraList, ifapply):
    result = []
    start = time.time()
    for i in range(len(paraList)):
        tn_tmp, _ = get_Ham_Pauli_SVD_v1(n, g, J, l, paraList, derive_TN=i)
        Pauli_list = generate_non_zero_Pauli(n, l)
        # Pauli_list = [generate_pauli_str(i, n) for i in range(4**n)]·
        with open("test.txt", "w") as f:
            for st in Pauli_list:
                val = get_Ham_Pauli_SVD_v2(tn_tmp, st, contract_flag=0, derive_TN=i).real / 2 ** n
                if abs(val) > 0.00001:
                    res = str(val) + " " + st.replace('', ' ').strip()
                    f.write(res + "\n")
        end = time.time()
        result.append(jl.calc_energy("../../test.txt", ifapply))
    time2 = (end - start)
    # print('n = ', n, "der time = ", time2)
    return result


def updateTheta(n, g, J, l, paraList, h, ifapply):
    # der = compute_derivative(n, g, J, l, paraList, ifapply)
    diff = compute_difference(n, g, J, l, paraList, ifapply)
    # print(der)
    # print(diff)
    for i in range(len(paraList)):
        paraList[i] -= h * diff[i]


def decayed_learning_rate(h, decay_rate, global_step, decay_step):
    return h * np.power(decay_rate, (global_step / decay_step))


def cal_energy(n, g, J, l, paraList, ifapply):
    tn_tmp, _ = get_Ham_Pauli_SVD_v1(n, g, J, l, paraList, derive_TN=None)
    Pauli_list = generate_non_zero_Pauli(n, l)
    # Pauli_list = [generate_pauli_str(i, n) for i in range(4**n)]
    with open("test.txt", "w") as f:
        for st in Pauli_list:
            val = get_Ham_Pauli_SVD_v2(tn_tmp, st, contract_flag=0, derive_TN=None).real / 2 ** n
            if abs(val) > 0.00001:
                res = str(val) + " " + st.replace('', ' ').strip()
                f.write(res + "\n")
    return jl.calc_energy("../../test.txt", ifapply)


def optimize_theta(n, g, J, l, paraList, loop, epsi, ifapply):
    pre_energy = cal_energy(n, g, J, l, paraList, ifapply)
    energy = 999
    for i in range(loop):
        if abs(energy - pre_energy) < epsi:
            return energy, i
        updateTheta(n, g, J, l, paraList, decayed_learning_rate(0.1, 0.98, i, 1000), ifapply)
        energy = cal_energy(n, g, J, l, paraList, ifapply)
        print(i, energy)
        pre_energy = energy
    return energy, loop


def main_loop():
    n = 3  # qubits
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
    pre_TN = 999
    pre_quantum = 999
    print(cal_energy(n, g, J, l, theta, 0))
    for i in range(20):
        print('------- loop', i, '--------')
        energy_TN = optimize_theta(n, g, J, l, theta, 500, 1e-5, i)
        energy_quantum = optimize_energy(n, g, J, l, theta)
        if abs(energy_TN[0] - pre_TN) + abs(energy_quantum[0] - pre_quantum) < 1e-5:
            print("Final energy & overall loop: ", energy_quantum[0], " ", i)
            return
        else:
            print("TN: ", energy_TN)
            print('quantum :', energy_quantum)
            print('theta :', theta)
            pre_TN = energy_TN[0]
            pre_quantum = energy_quantum[0]


if __name__ == '__main__':
    main_loop()
