from qiskit import *
import numpy as np

#Operator Imports
from qiskit.opflow import Z, X, I

#Circuit imports
from qiskit_nature.drivers import PySCFDriver, UnitsType, QMolecule, FermionicDriver
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.circuit.library import HartreeFock, UCCSD, UCC
from qiskit_nature.transformers import FreezeCoreTransformer, ActiveSpaceTransformer
from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit_nature.results import EigenstateResult
from qiskit import Aer
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.algorithms.optimizers import L_BFGS_B, SPSA, AQGD, CG, ADAM, P_BFGS, SLSQP, NELDER_MEAD
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.circuit.library import TwoLocal, EfficientSU2
import matplotlib.pyplot as plt
import matplotlib
from qiskit.tools.visualization import circuit_drawer
matplotlib.use('Agg')


driver = PySCFDriver(atom='H .0, .0, .0; Li .0, .0, 3;',
                     unit=UnitsType.ANGSTROM,
                     basis='sto3g')

at = ActiveSpaceTransformer(2, 2, active_orbitals=[0,4])
ft = FreezeCoreTransformer()

problem = ElectronicStructureProblem(driver, transformers=[ft, at])

# generate the second-quantized operators
second_q_ops = problem.second_q_ops()
main_op = second_q_ops[0]

num_particles = (problem.molecule_data_transformed.num_alpha,
                 problem.molecule_data_transformed.num_beta)

num_spin_orbitals = 2 * problem.molecule_data.num_molecular_orbitals
mapper = JordanWignerMapper()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)
qubit_op = converter.convert(main_op, num_particles=num_particles)
init_state = HartreeFock(num_spin_orbitals, num_particles, converter)

# Quantum circuit
# circ = UCCSD(qubit_converter=converter, num_particles=num_particles, num_spin_orbitals=8, reps=1, initial_state=init_state)
circ = EfficientSU2(num_qubits=num_spin_orbitals, entanglement='linear', reps=4)
circ.compose(init_state, front=True, inplace=True)
backend = Aer.get_backend('aer_simulator')
# backend.set_options(device='GPU')

#get results
optimizers = [L_BFGS_B()]
#ADAM(maxiter=10, beta_1=0.8, beta_2=0.8, amsgrad=True)
converge_cnts = np.empty([len(optimizers)], dtype=object)
converge_vals = np.empty([len(optimizers)], dtype=object)
converge_pars = np.empty([len(optimizers)], dtype=object)

for i, optimizer in enumerate(optimizers):
   counts = []
   values = []
   params = []
   def store_intermediate_result(eval_count, parameters, mean, std):
      counts.append(eval_count)
      values.append(mean)
      print(eval_count, mean)
      params.append(parameters)
   
   algorithm = VQE(circ,
            optimizer=optimizer,
            callback=store_intermediate_result,
            quantum_instance=backend)
   result = algorithm.compute_minimum_eigenvalue(qubit_op)
   print(result.eigenvalue.real)

   converge_cnts[i] = np.asarray(counts)
   converge_vals[i] = np.asarray(values)
   converge_pars[i] = np.asarray(params)
   # vqe_state = result.eigenstate
   #fidelity = state_fidelity(res_state, vqe_state)
   #print(fidelity)
numpy_solver = NumPyMinimumEigensolver()
calc = GroundStateEigensolver(converter, numpy_solver)
res_ref = calc.solve(problem)
res_ref = res_ref.eigenenergies.min()
print(res_ref)
plt.plot([0, 180],[res_ref, res_ref], label='reference energy')

for i, optimizer in enumerate(optimizers):
   plt.plot(converge_cnts[i], converge_vals[i], label=type(optimizer).__name__)
   plt.xlabel('Eval count')
   plt.ylabel('Energy')
   plt.title('Energy convergence for various optimizers using the UCC ansatz')
   plt.legend(loc='upper right')

plt.savefig('./result.png')