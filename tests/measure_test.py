from __future__ import annotations
from src.ops import *
from src.qubit import *
from src.solvers import *

np.random.seed(1722)

psi = state(alpha=np.random.random())
# print(psi)
psi, outcome, counts, _ = measure(psi, 100)
# print(counts)

# print(100 - 2 * np.sum(outcome) / 100)

angles = np.array([2.0, 3.0])

angle1 = angles[0] - 1.5
# print(angle1)
# print(angles[0])

state, _, _, _ = ansatz_1qubit(np.pi / 2, np.pi / 2)
print(state)
H, _ = hamiltonian_1qubit(2, e0=0.0, e1=4.0, V11=3.0, V12=0.2, V21=0.2, V22=-3, lam=1.0)
print(H)
print(measure_energy_1q(shots=10000))

print(np.abs(state.conj().T @ H @ state))
