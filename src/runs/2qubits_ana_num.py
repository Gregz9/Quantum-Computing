from __future__ import annotations
import numpy as np
import math
from src.solvers import *
from src.ops import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm

H = hamiltonian_2qubit(1.0, 2.0, 3.0, np.array([0.0, 2.5, 6.5, 7.0]))
lmbds = np.linspace(0.0, 1.0, 50)
eigvals_an = np.zeros((len(lmbds), H.shape[1]))
eigvals_num1 = np.zeros((len(lmbds), H.shape[1]))
eigvals_num2 = np.zeros((len(lmbds), H.shape[1]))
entropy = np.zeros((len(lmbds), H.shape[1]))

for i, lmbd in enumerate(lmbds):
    H = hamiltonian_2qubit(lmbd, 2.0, 3.0, np.array([0.0, 2.5, 6.5, 7.0]))
    eigvals, eigvecs = np.linalg.eig(H)
    eig_num1, _ = power_iteration(H, 400)
    eig_num2 = QR_solver(H)
    perm_an = eigvals.argsort()
    perm_num1 = eig_num1.argsort()
    perm_num2 = eig_num2.argsort()
    eigvals_an[i] = eigvals[perm_an]
    eig_num1 = eig_num1[perm_num1]
    eig_num2 = eig_num2[perm_num2]
    eigvals_num1[i] = eig_num1
    eigvals_num2[i] = eig_num2
    eigvecs = eigvecs[:, perm_an]
    entropy[i] = von_neumann(eigvals, eigvecs, 0)

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
for i in range(4):
    axs.plot(lmbds, eigvals_an[:, i], label=f"$E_{i}$")
axs.set_xlabel(r"$\lambda$")
axs.set_ylabel("Energy")
axs.legend()
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
for i in range(4):
    axs.plot(lmbds, entropy[:, i], label=f"$H_{i}$")
axs.set_xlabel(r"$\lambda$")
axs.set_ylabel("Entropy")
axs.legend()
plt.show()

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
