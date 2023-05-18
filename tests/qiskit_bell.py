import numpy as np
import qiskit as qk
from scipy.optimize import minimize


# Initialize registers and circuit
n_qubits = 1  # Number of qubits
n_cbits = 1  # Number of classical bits (the number of qubits you want to measure at the end of the circuit)
qreg = qk.QuantumRegister(n_qubits)  # Create a quantum register
creg = qk.ClassicalRegister(n_cbits)  # Create a classical register
circuit = qk.QuantumCircuit(qreg, creg)  # Create your quantum circuit
circuit.draw()  # Draw circuit. It is empty
# Perform operations on qubit
circuit.x(qreg[0])  # Applies a Pauli X gate to the first qubit in the quantum register
circuit.draw()
# Chose a qubit to measure and encode the results to a classical bit
# Measure the first qubit in the quantum register
# and encode the results to the first qubit in the classical register
circuit.measure(qreg[0], creg[0])
circuit.draw()
# Execute circuit

backend = qk.Aer.get_backend("qasm_simulator")
# This is the device you want to use. It is an ideal simulation of a quantum device


job = backend.run(circuit, shots=1000)  # Run the circuit 1000 times
result = job.result()
counts = result.get_counts()
print(counts)
circuit.clear()
circuit.draw()

circuit.h(qreg[0])  # Apply a Hadamard gate to the first qubit of the quantum register
circuit.measure(qreg, creg)
print(circuit.draw())
job = backend.run(circuit, shots=1000)
result = job.result()
counts = result.get_counts()
print(counts)
circuit.clear()

n_qubits = 2
n_cbits = 2
qreg = qk.QuantumRegister(n_qubits)
creg = qk.ClassicalRegister(n_cbits)
circuit = qk.QuantumCircuit(qreg, creg)
circuit.draw()
circuit.h(qreg[0])
circuit.cx(qreg[0], qreg[1])


circuit.draw()
circuit.measure(qreg, creg)
circuit.draw()
job = backend.run(circuit, shots=1000)
result = job.result()
counts = result.get_counts()
print(counts)
circuit.clear()
