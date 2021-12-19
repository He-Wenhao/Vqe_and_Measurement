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


def optimize_energy(n, g, J, l, paraList, loop):
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
    res = jl.optimize("../../test.txt", loop)
    #        print('n = ', n, "energy time = ", time2)
    return res


def compute_difference(n, g, J, l, paraList, ifapply):
    result = []
    for i in range(len(paraList)):
        pre_energy = cal_energy(n, g, J, l, paraList, ifapply)
        paraList[i] += 1e-7
        energy = cal_energy(n, g, J, l, paraList, ifapply)
        diff = (energy - pre_energy) / 1e-7
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


def updateTheta(n, g, J, l, paraList, step, h, ifapply):
    # Adam
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = [0. for _ in range(len(paraList))]
    v = [0. for _ in range(len(paraList))]
    grad = compute_difference(n, g, J, l, paraList, ifapply)
    for i in range(len(paraList)):
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] ** 2
        mhat = m[i] / (1.0 - beta1 ** (step + 1))
        vhat = v[i] / (1.0 - beta2 ** (step + 1))
        paraList[i] = paraList[i] - h * mhat / (sqrt(vhat) + eps)


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


def optimize_theta(n, g, J, l, paraList, loop, epsi, ifapply, pre_energy=999):
    energy = 999
    h = 0.2 * (0.5 ** ifapply)
    for i in range(loop):
        updateTheta(n, g, J, l, paraList, i, decayed_learning_rate(h, 0.5, i, 50), ifapply)
        energy = cal_energy(n, g, J, l, paraList, ifapply)
        if pre_energy - energy < 0:
            h *= 0.7
            print("learning rate: ", h)
        elif pre_energy - energy < epsi:
            print(i, ": ", energy)
            return energy, i
        print(i, ": ", energy)
        pre_energy = energy
    return energy, loop


def main_loop(num):
    n = num  # qubits
    g = 10  # parameter of Ising
    J = 0.1  # parameter of Ising
    l = 1  # layers
    ## for debug, accurate diagonalization
    ideal_energy = spectrum_1dTransverse_Ising(n, g, J)[0]
    print("ideal: ", ideal_energy)
    # parameter for TN
    n_theta = UniTN_1d_para_cnt(n, l, 3)  # nums of parameters
    theta = [random.uniform(0, np.pi * 2) for i in range(n_theta)]
    pre_quantum = 999
    print(cal_energy(n, g, J, l, theta, 0))
    print("--------TN0--------")
    energy_TN = optimize_theta(n, g, J, l, theta, 200, 1e-3, 0, pre_quantum)
    print("TN0: ", energy_TN)
    print("--------VQE--------")
    energy_quantum = optimize_energy(n, g, J, l, theta, 500)
    print('quantum :', energy_quantum)
    print("--------TN1--------")
    energy_TN = optimize_theta(n, g, J, l, theta, 200, 1e-3, 1, pre_quantum)
    print("TN1: ", energy_TN)
    '''if energy_quantum[0] > energy_TN[0]:
        print("Final energy & overall loop: ", min(energy_TN[0], energy_quantum[0]))
        return
    else:
        pre_quantum = energy_quantum[0]'''


def vqe_only(num):
    n = num  # qubits
    g = 10  # parameter of Ising
    J = 0.1  # parameter of Ising
    l = 1  # layers
    n_theta = UniTN_1d_para_cnt(n, l, 3)  # nums of parameters
    theta = [random.uniform(0, np.pi * 2) for i in range(n_theta)]
    print("vqe only: ", optimize_energy(n, g, J, l, theta, 500))


if __name__ == '__main__':
    n = 14
    vqe_only(n)
    main_loop(n)
