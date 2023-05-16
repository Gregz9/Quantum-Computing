from __future__ import annotations
import numpy as np
import math
from src.solvers import *
from src.ops import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm

basis0 = np.array([1, 0])
basis1 = np.array([0, 1])

op0 = np.kron(basis0, Identity())
op1 = np.kron(basis1, Identity())

H = hamiltonian_2qubit(1.0, 2.0, 3.0, np.array([0.0, 2.5, 6.5, 7.0]))
# print(H)
lmbds = np.linspace(0.0, 1.0, 50)
eigvals_an = np.zeros((len(lmbds), H.shape[1]))
entropy = np.zeros((len(lmbds), H.shape[1]))

for i, lmbd in enumerate(lmbds):
    H = hamiltonian_2qubit(lmbd, 2.0, 3.0, np.array([0.0, 2.5, 6.5, 7.0]))
    eigvals, eigvecs = np.linalg.eig(H)
    perm = eigvals.argsort()
    eigvals_an[i] = eigvals[perm]
    eigvecs = eigvecs[:, perm]
    entropy[i] = von_neumann(eigvals, eigvecs, 0)

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
for i in range(4):
    axs.plot(lmbds, entropy[:, i], label=f"$H_{i}$")
axs.set_xlabel(r"$\lambda$")
axs.set_ylabel("Entropy")
axs.legend()
plt.show()

#
# eig_vals, _ = np.linalg.eig(H)
# perm = eig_vals.argsort()
# print(eig_vals[perm])
# eig_vals, _ = power_iteration(H, 400)
# perm = eig_vals.argsort()
# print(eig_vals[perm])
#
#
# eig_vals = QR_solver(H)
# perm = eig_vals.argsort()
# print(eig_vals[perm])
#
# state = np.random.random(size=4)
# print(
#     np.kron(np.array([0.0, 1.0]), Identity())
#     @ np.kron(np.array([0.0, 1.0]), Identity()).T
# )
# print(np.outer(state, state))
# print(trace_first(state))
# print(trace_second(state))
