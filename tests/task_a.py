from __future__ import annotations
import numpy as np
import math
from src.solvers import *
from src.ops import *
from src.bell_states import *
import qiskit as qk
from scipy.optimize import minimize

if __name__ == "__main__":
    # Setting up the one qubit basis
    psi0 = state(alpha=1.0)
    psi1 = state(beta=1.0)

    X_psi0 = PauliX() @ psi0
    X_psi1 = PauliX() @ psi1

    Y_psi0 = PauliY() @ psi0
    Y_psi1 = PauliY() @ psi1

    Z_psi0 = PauliZ() @ psi0
    Z_psi1 = PauliZ() @ psi1

    H_psi0 = Hadamard() @ psi0
    H_psi1 = Hadamard() @ psi1

    # Qiskit version
    n_qubits = 1
    n_cbits = 1
    qreg = qk.QuantumRegister(n_qubits)
    creg = qk.ClassicalRegister(n_cbits)
    circuit = qk.QuantumCircuit(qreg, creg)
    print(circuit.draw())

    circuit.x(qreg[0])
    print(circuit.draw())

    circuit.measure(qreg[0], creg[0])
    print(circuit.draw())

    backend = qk.Aer.get_backend("qasm_simulator")
    job = backend.run(circuit, shots=1000)
    result = job.result()
    counts = result.get_counts()
    print(counts)
    #
    # circuit.clear()
    # circuit.draw()
    #
    # circuit.h(qreg[0])
    # circuit.measure(qreg, creg)
    # print(circuit.draw())
    # job = backend.run(circuit, shots=1000)
    # result = job.result()
    # counts = result.get_counts()
    # print(counts)
    # circuit.clear()
    # print(np.hstack((q0,q1)))

    # psi = bell_state(0, 0)
    # alpha = np.random.random() / 2.0 + 1.0j * np.random.random() / 2.0
    # psi = qubit(alpha=alpha)
    #
    # Hdag = Hadamard().conj().T
    # CnotDag = Cnot().conj().T
    #
    # R_psi = np.kron(Hdag, Identity()) @ CnotDag @ psi
    # print(R_psi)
    #
    # prob0, state0 = Measure(psi,0,1)
    # print(prob0)
    # print(state0)
    #
    # prob1, state1 = Measure(state0, 1, 1)
    # print(prob1)
    # print(state1)
    #
    # prob0, state0 = Measure(psi,1,0)
    # print(prob0)
    # print(state0)
    #
    # prob1, state1 = Measure(state0, 0, 0)
    # print(prob1)
    # print(state1)
