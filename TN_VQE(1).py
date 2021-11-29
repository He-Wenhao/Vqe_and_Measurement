from qiskit import *
import numpy as np

#Operator Imports
from qiskit.opflow import Z, X, I

#Circuit imports
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.circuit.library import HartreeFock, UCCSD, UCC
from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit_nature.results import EigenstateResult
from qiskit import Aer
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.algorithms.optimizers import L_BFGS_B, SPSA, AQGD, CG, ADAM, P_BFGS, SLSQP, NELDER_MEAD
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.circuit.library import TwoLocal, EfficientSU2
import matplotlib.pyplot as plt
import matplotlib, pickle
from qiskit.tools.visualization import circuit_drawer
import qiskit.quantum_info as qi
from  qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.quantum_info.operators import Operator, Pauli
import read_hamiltonian
matplotlib.use('Agg')



def Ansatz_optimization(num_qubits, is_first_step):
    r"""
    Given the Hamiltonian in the qubit form. Initalize a hardware efficient ansatz and optimize the ansatz.
    Args:
        num_qubits: number of qubits requested for the model.
        qubit_op: the given qubit form Hamiltonian read from outside
    Save/ Returns:
        op_state: optimized state from the ansatz.
        The optimal circuit during the trajectory of the VQE optimization.
        The image of the process of optimization.
    """
    qubit_op = read_hamiltonian.read(num_qubits=num_qubits)
    optimizer = SPSA()
    # Quantum circuit
    ansatz = EfficientSU2(num_qubits=num_qubits, entanglement='linear', reps=4)
    backend = Aer.get_backend('aer_simulator')
    # backend.set_options(device='GPU')

    counts = []
    values = []
    params = []
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)
        params.append(parameters)

    if is_first_step:
        algorithm = VQE(ansatz,
                optimizer=optimizer,
                callback=store_intermediate_result,
                quantum_instance=backend)
    else:
        output = open('optimal_point.pkl', 'wb')
        optimal_point = pickle.load(output)[0]
        output.close()
        algorithm = VQE(ansatz,
                optimizer=optimizer,
                initial_point=optimal_point,
                callback=store_intermediate_result,
                quantum_instance=backend)
    result = algorithm.compute_minimum_eigenvalue(qubit_op)
    print('vqe computed energy is', result.eigenvalue.real)


    # save the optimal circuit
    optimal_point = result.optimal_point
    # optimal_circ = ansatz.bind_parameters(optimal_point)
    # op_state = qi.Statevector(optimal_circ)
    # circ_matrix = qi.DensityMatrix(optimal_circ)
    # circ_matrix = circ_matrix.data
    # matrix_op = qubit_op.to_matrix()
    output = open('optimal_point.pkl', 'wb')
    pickle.dump(optimal_point, output)
    output.close()


    plt.plot(counts, values)
    plt.xlabel('Eval count')
    plt.ylabel('Energy')
    plt.title('Energy convergence for the given Hamiltonian and hardware-efficient ansatz')
    plt.legend(loc='upper right')

    plt.savefig('./result.png')

    return result.eigenvalue.real


