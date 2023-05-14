from __future__ import annotations
import numpy as np
import math
from src.solvers import *
from src.ops import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Solving the Hamiltonian analytically

lam = np.arange(0, 1, 0.05)
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

theta = np.pi
phi = np.pi

measure_energy_1q(theta, phi, lam[0], shots=1)

# figs, axs = plt.subplots(1, 1, figsize=(12, 12))
# for j in range(2):
#     axs.plot(lam, system_ener_an[:, j], label=f"$E_{j+1}$")
#     axs.plot(lam, system_ener_num[:, j], label=f"$E_{j+1}$")
# axs.set_xlabel(r"$\lambda$")
# axs.set_label("Energy")
# axs.grid()
# axs.legend()
# plt.show()
