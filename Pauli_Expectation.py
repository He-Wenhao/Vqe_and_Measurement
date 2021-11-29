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


def if_non_zero_Pauli(P_str, l):
    n = len(P_str)
    if P_str == "I" * n:
        return False
    for i in range(n):
        if P_str[i] != "I":
            for j in range(n):
                if P_str[n - 1 - j] != "I":
                    if n - 1 - j - i >= 2 + 2 * l:
                        return False
                    else:
                        return True


def generate_non_zero_Pauli(n, l):
    res_list = []
    length = 2 + 2 * l
    if n % 2 == 0:
        for i in range(n - length):
            if i != 1 and i != n - length - 1 and i != (n-length)/2:
                for j in range(4 ** length):
                    res_list.append("I" * i + generate_pauli_str(i, length) + "I" * (n - length - i))
    else:
        for i in range(n - length):
            if i != 1 and i != n - length - 1:
                for j in range(4 ** length):
                    res_list.append("I" * i + generate_pauli_str(i, length) + "I" * (n - length - i))
    return res_list


def pauli_trace(d_matrix, n, l):
    with open("test.txt", "w") as f:
        for i in range(4 ** n):
            st = generate_pauli_str(i, n)
            if if_non_zero_Pauli(st, l):
                pauli_op = qi.Operator.from_label(st)
                val = d_matrix.expectation_value(pauli_op).real
                res = str(val) + " " + st.replace('', ' ').strip()
                f.write(res + "\n")
