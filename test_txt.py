from qiskit import *
import numpy as np

#Operator Imports
# from qiskit.opflow.operator_globals import Y

#Circuit imports
from qiskit import Aer
from qiskit.tools.visualization import circuit_drawer
import qiskit.quantum_info as qi
from  qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.quantum_info.operators import Operator, Pauli
from functools import reduce


## Pauli Matrices
PX = np.array([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]])
PY = np.array([[0.+0.j, 0. - 1.j], [0. + 1.j, 0.+0.j]])
PZ = np.array([[1.+0.j, 0.+0.j], [0.+0.j, -1.+0.j]])
Id = np.array([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]])

def test_txt(num_qubits):
    with open("test.txt", "r") as f:
        data = f.readlines()
        for i in range(len(data)):
            # Split the string with space
            str_list = data[i].split()
            for j in range(num_qubits):
                if str_list[j+1] == 'I':
                    str_list[j+1] = Id
                elif str_list[j+1] == 'X':
                    str_list[j+1] = PX
                elif str_list[j+1] == 'Y':
                    str_list[j+1] = PY
                else:
                    str_list[j+1] = PZ
            if i == 0:
                qubit_op = complex(str_list[0]) * reduce(lambda x, y: np.kron(x,y), str_list[1:])
            else:
                qubit_op += complex(str_list[0]) * reduce(lambda x, y: np.kron(x,y), str_list[1:])
    
    return qubit_op


