from __future__ import annotations
import numpy as np
from src.qubit import *
from src.ops import *
from src.bell_states import *
from tut import state
import scipy
import itertools


alpha = np.random.random()
beta = np.random.random()
delta = np.random.random()

basic = qubit(alpha=1.0)
psi = qubit(alpha=alpha)
phi = qubit(beta=beta)
xi = qubit(np.random.random())
two_state = np.kron(psi, phi)


# Setting up the gates/matrices
_PAULI_X = PauliX()
_PAULI_Y = PauliY()
_PAULI_Z = PauliZ()
_HADAMARD = Hadamard()
_SGATE = Sgate()
_TGATE = Tgate()
# print(bell_state(1, 1))
prob, m_psi = Measure(psi, idx=0, tostate=1)
print(m_psi)
print(prob)
# print(psi)
# print("\n")
# print(_PAULI_X)
# print(_PAULI_X @ psi)
# print("\n")
# print(_PAULI_Y)
# print(_PAULI_Y @ psi)
# print("\n")
# print(_PAULI_Z)
# print(_PAULI_Z @ psi)
# print("\n")
# print(_HADAMARD)
# H_psi = _HADAMARD @ psi
# S_psi = _SGATE @ psi
# T_psi = _TGATE @ psi
#
# dump(T_psi)
