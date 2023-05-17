from __future__ import annotations
import numpy as np
import math
from scipy.optimize import minimize
from tqdm import tqdm
from src.solvers import *
from src.ops import *

basis0 = np.array([1.0, 0.0])
basis00 = np.kron(basis0, basis0)
basis10 = np.array([0.0, 1.0, 0.0, 0.0])
basis01 = np.array([0.0, 0.0, 1.0, 0.0])
basis11 = np.array([0.0, 0.0, 0.0, 1.0])

# Testing if the initialization of a two qubit ansatz works correctly
state = ansatz_2qubit()
print(state)

Z = np.array([[1.0, 0.0], [0.0, -1.0]])
I = np.eye(2)
II = np.kron(I, I)
IZ = np.kron(I, Z)
ZI = np.kron(Z, I)
ZZ = np.kron(Z, Z)

# Our goal is to rewrite the hamiltonian of the system in terms of the Pauli Z matrix, and
# the identity matrix. Using this two projection operators below, we can write four equation
# which we will simplify, which turn out to be equivalent to the standard projection operators.
P_0 = (1 / 2) * (I + Z)
P_1 = (1 / 2) * (I - Z)

assert np.allclose(np.outer(basis00, basis00), (1 / 4) * (II + ZI + IZ + ZZ))
assert np.allclose(np.outer(basis10, basis10), (1 / 4) * (II + ZI - IZ - ZZ))
assert np.allclose(np.outer(basis01, basis01), (1 / 4) * (II - ZI + IZ - ZZ))
assert np.allclose(np.outer(basis11, basis11), (1 / 4) * (II - ZI - IZ + ZZ))

# However, instead of writing each of those by them selves, we are going to set
# up a 4x4 matrix, and compute four coefficients A,B,C and D in order to rewrite
# the Hamiltonian using tensor products consisting of I and Z.
# print(IZ)
# print(Swap() @ IZ @ Swap())

X = PauliX()
XX = np.kron(X, X)
H = Hadamard()
HH = np.kron(H, H)
# print(XX)

measure_xx = Cnot(0, 1) @ HH @ state
print(state**2)
print(state.conj() * state)
print(measure_xx.conj() * measure_xx)

measure_zz = Cnot() @ state
print(measure_zz**2)
print(measure_zz.conj() * measure_zz)
# ones = np.ones(4) * 0.25
# print(ZI @ ones)
