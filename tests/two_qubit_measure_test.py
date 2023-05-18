from __future__ import annotations
import numpy as np
from src.solvers import *
from src.ops import *

state = ansatz_2qubit()

exp_val = measure_energy_2q()
print(exp_val)
H = hamiltonian_2qubit(1.0, 2.0, 3.0, [0.0, 2.5, 6.5, 7.0])
print(state.conj().T @ H @ state)
