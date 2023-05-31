from __future__ import annotations
import numpy as np
import math
from src.solvers import *
from src.ops import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

H = hamiltonian_2qubit(1.0, 2.0, 3.0, np.array([0.0, 2.5, 6.5, 7.0]))
lmbds = np.linspace(0.0, 1.0, 50)
eigvals_an = np.zeros((len(lmbds), H.shape[1]))
entropy = np.zeros((len(lmbds), H.shape[1]))

start_time = time.time()
for i, lmbd in enumerate(lmbds):
    H = hamiltonian_2qubit(lmbd, 2.0, 3.0, np.array([0.0, 2.5, 6.5, 7.0]))
    eigvals, eigvecs = np.linalg.eig(H)
    perm_an = eigvals.argsort()
    eigvals_an[i] = eigvals[perm_an]
    eigvecs = eigvecs[:, perm_an]
    entropy[i] = von_neumann(eigvals, eigvecs, 0)
print(f"Time taken for the analytical method: {time.time() - start_time}")

start_time = time.time()
eigvals_num1 = np.zeros((len(lmbds), H.shape[1]))
for i, lmbd in enumerate(lmbds):
    H = hamiltonian_2qubit(lmbd, 2.0, 3.0, np.array([0.0, 2.5, 6.5, 7.0]))
    eig_num1, _ = power_iteration(H, 400)
    perm_num1 = eig_num1.argsort()
    eig_num1 = eig_num1[perm_num1]
    eigvals_num1[i] = eig_num1
print(f"Time taken for the numerical method: {time.time() - start_time}")

start_time = time.time()
learning_rate = 0.2
momentum = 0.2
num_shots = 500
max_epochs = 5000
lmbds2 = np.linspace(0.0, 1.0, 11)
min_energy = np.zeros(len(lmbds2))
epochs = np.zeros(len(lmbds))
for i, lmbd in enumerate(tqdm(lmbds2)):
    angles = np.random.uniform(0.0, np.pi, size=4)
    angles, epoch, min_energy[i], delta_energy = VQE_momentum(
        learning_rate, momentum, max_epochs, num_shots, angles, lmbd
    )
# print(
#     f"Lambda: {lmbds[i]}, Energy: {energy}, Epochs: {epoch}, Delta_energy : {delta_energy}"
# )
print(f"Time taken for the VQE method: {time.time() - start_time}")

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
for i in range(4):
    axs.plot(lmbds, eigvals_an[:, i], label=f"$E_{i}$")
axs.scatter(lmbds2, min_energy, label=f"$E_0")
axs.set_xlabel(r"$\lambda$")
axs.set_ylabel("Energy")
axs.set_title(
    "Energy levels of the two-body system as a function of interatction strength $\lambda$"
)
axs.legend()
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
for i in range(4):
    axs.plot(lmbds, eigvals_num1[:, i], label=f"$E_{i}$")
axs.scatter(lmbds2, min_energy, label=f"$E_0")
axs.set_xlabel(r"$\lambda$")
axs.set_ylabel("Energy")
axs.set_title(
    "Energy levels of the two-body system as a function of interatction strength $\lambda$"
)
axs.legend()
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
for i in range(1):
    axs.plot(lmbds, entropy[:, i], label=f"$H_{i}$")
axs.set_xlabel(r"$\lambda$")
axs.set_ylabel("Entropy")
axs.set_title(
    "Von Neumann Entropy of a the two-body system as a function of interaction strength $\lambda$"
)
axs.legend()
plt.show()
