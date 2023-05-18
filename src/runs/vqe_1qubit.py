from __future__ import annotations
import numpy as np
import math
from src.solvers import *
from src.ops import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm

# Solving the Hamiltonian analytically


def plot_energies(analytical, numerical=None, lmd=np.zeros(20)):
    figs, axs = plt.subplots(1, 1, figsize=(12, 12))
    for j in range(2):
        axs.plot(lmd, analytical[:, j], label=f"$E_{j+1}$")
        # axs.plot(lmd, numerical[:, j], label=f"$E_{j+1}$")
    axs.set_xlabel(r"$\lambda$")
    axs.set_label("Energy")
    axs.grid()
    axs.legend()
    plt.show()


def plot_vqe_energies(energies, analytical, lmbds):
    figs, axs = plt.subplots(1, 1, figsize=(12, 12))
    for i in range(2):
        axs.plot(lmbds, analytical[:, i], label=f"$E_{i+1}$")
    axs.scatter(lmbds, energies, label=f"$E_0$")
    axs.set_xlabel(r"$\lambda$")
    axs.set_label("Energy")
    axs.grid()
    axs.legend()
    plt.show()


# lam = np.arange(0, 1, 0.05)
lam = np.linspace(0.0, 1.0, 40)
system_ener_an = np.zeros((len(lam), 2))
for i, l in enumerate(lam):
    Ham, _ = hamiltonian_1qubit(
        2, e0=0.0, e1=4.0, V11=3, V12=0.2, V21=0.2, V22=-3, lam=l
    )
    eig_vals, _ = np.linalg.eig(Ham)
    perm = eig_vals.argsort()
    system_ener_an[i] = eig_vals[perm]

# Solving the Hamiltonian numerically

system_ener_num = np.zeros((len(lam), 2))
for i, l in enumerate(lam):
    Ham, _ = hamiltonian_1qubit(
        2, e0=0.0, e1=4.0, V11=3, V12=0.2, V21=0.2, V22=-3, lam=l
    )
    eig_vals, _ = power_iteration(Ham, 100)
    eig_vals = np.where(np.isnan(eig_vals), 0, eig_vals)
    perm = eig_vals.argsort()
    system_ener_num[i] = eig_vals[perm]

# Solving for the eigenvalalues using the VQE

# num_shots = 1000
# learning_rate = 0.3
# max_epochs = 400
# lmbds = np.linspace(0.0, 1.0, 40)
# min_energy = np.zeros(len(lmbds))
# epochs = np.zeros(len(lmbds))
# for i, lmbd in enumerate(tqdm(lmbds)):
#     init_angles = np.random.uniform(0, np.pi, size=2)
#     angles, epochs[i], energy, delta_energy = VQE_1qubit(
#         learning_rate, max_epochs, num_shots, init_angles, lmbd
#     )
#     if epochs[i] < (epochs[i - 1] - 5):
#         init_angles = np.random.uniform(0, np.pi, size=2)
#         angles, epochs[i], energy, delta_energy = VQE_1qubit(
#             learning_rate, max_epochs, num_shots, init_angles, lmbd
#         )
#     min_energy[i] = measure_energy_1q(angles, lmbd, num_shots)

num_shots = 1000
learning_rate = 0.3
momentum = 0.2
max_epochs = 200
lmbds = np.linspace(0.0, 1.0, 40)
min_energy = np.zeros(len(lmbds))
epochs = np.zeros(len(lmbds))
for i, lmbd in enumerate(tqdm(lmbds)):
    init_angles = np.random.uniform(0, np.pi, size=2)
    angles, epochs[i], energy, delta_energy = VQE_momentum(
        learning_rate, momentum, max_epochs, num_shots, init_angles, lmbd
    )
    if epochs[i] < (epochs[i - 1] - 5):
        init_angles = np.random.uniform(0, np.pi, size=2)
        angles, epochs[i], energy, delta_energy = VQE_momentum(
            learning_rate, momentum, max_epochs, num_shots, init_angles, lmbd
        )
    min_energy[i] = measure_energy_1q(angles, lmbd, num_shots)


print(epochs)
plot_vqe_energies(min_energy, system_ener_an, lmbds)
