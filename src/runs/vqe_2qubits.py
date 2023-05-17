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
state = ansatz_2qubit(np.pi / 2, np.pi / 2)
# print(state)

Z = np.array([[1.0, 0.0], [0.0, -1.0]])
I = np.eye(2)
IZ = np.kron(I, Z)
ZI = np.kron(Z, I)
ZZ = np.kron(Z, Z)
# print(IZ)
# print(np.kron(Z, I))
# print(np.kron(Z, Z))

P_0 = (1 / 2) * (I + Z)
P_1 = (1 / 2) * (I - Z)

dens_basis00 = np.kron(I, I) + ZI + IZ + ZZ
print(dens_basis00)

dens_basis01 = np.outer(basis01, basis01)
print(dens_basis01)
dens_basis01 = np.kron(I, I) - ZI + IZ - ZZ
print(dens_basis01)
