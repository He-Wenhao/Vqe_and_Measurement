import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
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
    for i in range(4 ** n):
        st = generate_pauli_str(i, n)
        pauli_op = qi.Operator.from_label(st)
        print(st + " : " + str(d_matrix.expectation_value(pauli_op)))


# eg 1
print("======== eg 1 ========")
rho_p = qi.DensityMatrix.from_label('-')
print(rho_p.tensor(rho_p))
pauli_trace(rho_p.tensor(rho_p), 2)

# eg 2
print("======== eg 2 ========")
qc_AB = QuantumCircuit(2)
qc_AB.h(0)
qc_AB.cx(0, 1)
qc_AB.draw()
rho_p = qi.DensityMatrix.from_instruction(qc_AB)
print(rho_p)
pauli_trace(rho_p, 2)
