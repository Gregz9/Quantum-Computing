from __future__ import annotations
import numpy as np
import math
from src.solvers import *
from src.ops import *
from src.bell_states import *
import qiskit as qk
from scipy.optimize import minimize

if __name__ == "__main__":
    # Preparing two seperate qubits each in one of the basis state |0> and |1>
    psi0 = state(alpha=1.0)
    psi1 = state(beta=1.0)

    # number of measurements
    shots = 1000

    print("Applying PauliX gate to state [1.0j, 0.0j]")
    print("----------------------------------------------------")
    probs, counts, measurement_probs, post_psi = measurement(psi0, PauliX(), shots)

    print(
        f"State of the qubit prior to applying pauli X matrix: \
                \n|0>: {psi0[0]}   |1>: {psi0[1]}\n"
    )
    print(
        f"State of the qubit after applying pauli X matrix: \
                \n|0>: {post_psi[0]}   |1>: {post_psi[1]}\n"
    )
    print(f"Performing {shots} measurements")
    print(
        f"Probabilty after applying pauli X gate: \
          \n|0> : {probs[0]*100}%  \
          \n|1> : {probs[1]*100}% \n"
    )
    print(
        f"Observed states: \
            \n|0> : {shots - counts[0]} \
            \n|1> : {counts[0]}\n"
    )
    print(
        f"Observed probabilities: \
            \n|0> : {100 - measurement_probs[0]*100}% \
            \n|1> : {measurement_probs[0] * 100}%  "
    )
    print("----------------------------------------------------\n")

    print("Applying PauliY gate to state [0.0j, 1.0j]")
    print("----------------------------------------------------")
    probs, counts, measurement_probs, post_psi2 = measurement(psi1, PauliY(), shots)
    print(
        f"State of the qubit prior to applying pauli Y matrix: \
                \n|0>: {psi1[0]}   |1>: {psi1[1]}\n"
    )
    print(
        f"State of the qubit after applying pauli Y matrix: \
                \n|0>: {post_psi2[0]}   |1>: {post_psi2[1]}\n"
    )
    print(f"Performing {shots} measurements")
    print(
        f"Probabilty after applying pauli Y gate: \
          \n|0> : {probs[0]*100}%  \
          \n|1> : {probs[1]*100}% \n"
    )
    print(
        f"Observed states: \
            \n|0> : {counts[0]}\
            \n|1> : {shots - counts[0]} \n"
    )
    print(
        f"Observed probabilities: \
            \n|0> : {measurement_probs[0] * 100}%  \
            \n|1> : {100 - measurement_probs[0]*100}% "
    )
    print("----------------------------------------------------\n")

    print("Applying PauliZ gate to state [0.0j, 1.0j]")
    print("----------------------------------------------------")
    probs, counts, measurement_probs, post_psi2 = measurement(psi1, PauliZ(), shots)
    print(
        f"State of the qubit prior to applying pauli Z matrix: \
                \n|0>: {psi1[0]}   |1>: {psi1[1]}\n"
    )
    print(
        f"State of the qubit after applying pauli Z matrix: \
                \n|0>: {post_psi2[0]}   |1>: {post_psi2[1]}\n"
    )
    print(f"Performing {shots} measurements")
    print(
        f"Probabilty after applying pauli Z gate: \
          \n|0> : {probs[0]*100}%  \
          \n|1> : {probs[1]*100}% \n"
    )
    print(
        f"Observed states: \
            \n|0> : {shots - counts[0]} \
            \n|1> : {counts[0]}\n"
    )
    print(
        f"Observed probabilities: \
            \n|0> : {100 - measurement_probs[0]*100}% \
            \n|1> : {measurement_probs[0] * 100}%  \n"
    )

    Z_psi0 = PauliZ() @ psi0
    Z_psi1 = PauliZ() @ psi1

    H_psi0 = Hadamard() @ psi0
    H_psi1 = Hadamard() @ psi1

    S_psi0 = Sgate() @ psi0
    S_psi1 = Sgate() @ psi1

    psi_test = state(alpha=np.random.random())

    # Qiskit version
    n_qubits = 1
    n_cbits = 1
    qreg = qk.QuantumRegister(n_qubits)
    creg = qk.ClassicalRegister(n_cbits)
    circuit = qk.QuantumCircuit(qreg, creg)

    circuit.x(qreg[0])

    circuit.measure(qreg[0], creg[0])
    print(circuit.draw())

    backend = qk.Aer.get_backend("qasm_simulator")
    job = backend.run(circuit, shots=1000)
    result = job.result()
    counts = result.get_counts()
    print(counts)
