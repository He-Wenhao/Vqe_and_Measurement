from HamTN import *
import qiskit.quantum_info as qi
import numpy as np


def generate_pauli_str(number, n):
    replace_dic = {"0": "I", "1": "X", "2": "Y", "3": "Z"}
    f_str = np.base_repr(number, base=4)
    if len(f_str) < n:
        f_str = "0" * (n - len(f_str)) + f_str
    for key in replace_dic.keys():
        f_str = f_str.replace(key, replace_dic[key])
    return f_str


def pauli_trace(d_matrix, n):
    with open("test.txt", "w") as f:
        for i in range(4 ** n):
            st = generate_pauli_str(i, n)
            pauli_op = qi.Operator.from_label(st)
            res = str(d_matrix.expectation_value(pauli_op).real) + " " + st.replace('', ' ').strip()
            f.write(res + "\n")


if __name__ == '__main__':
    n = 5  # qubits
    g = 10  # parameter of Ising
    J = 0.1  # parameter of Ising
    l = 2  # layers
    n_para = UniTN_1d_para_cnt(n, l)  # nums of parameters
    paraList = [random.uniform(1, 10) for i in range(n_para)]
    H = qi.DensityMatrix(get_Ham(n, g, J, l, paraList))  # get Hamiltonian matrix
    pauli_trace(H, n)
