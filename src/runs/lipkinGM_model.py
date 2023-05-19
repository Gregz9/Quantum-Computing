from __future__ import annotations
from src.solvers import *
from src.ops import *
from tqdm import tqdm
import matplotlib.pyplot as plt

# The case of spin J=1, solving for eigenvalues using analytical and numerical methods


def plot_energy(energy_an, energy_vqe, v_values_ana, v_values_vqe):
    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    for i in range(len(energy_an[0])):
        axs.plot(v_values_ana, energy_an[:, i], label=f"$E_{i}$")
    axs.set_xlabel(r"$V/\epsilon$")
    axs.scatter(v_values_vqe, energy_vqe, label="VQE Energy Scipy", marker="o")
    axs.set_ylabel(r"$E/\epsilon$")
    axs.legend(loc="upper left")
    plt.show()


def plot_energy_FH(energy_an, energy_an_FH, v_values_ana):
    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    for i in range(len(energy_an[0])):
        # axs.plot(v_values_ana, energy_an[:, i], label=f"$E_{i}$")
        axs.plot(v_values_ana, energy_an_FH[:, i], label=f"$E_FH{i}$")
    axs.set_xlabel(r"$V/\epsilon$")
    axs.set_ylabel(r"$E/\epsilon$")
    axs.legend(loc="upper left")
    plt.show()


Vs_an = np.linspace(0.0, 2.0, 200)
eigvals_an = np.zeros((len(Vs_an), 3))
eigvals_num_PI = np.zeros((len(Vs_an), 3))
eigvals_num_QR = np.zeros((len(Vs_an), 3))

for i, v in enumerate(tqdm(Vs_an)):
    H = lipkin_H_J1(v)
    eigvals, eigvecs = np.linalg.eig(H)
    eig_num_PI, _ = power_iteration(H, 1000)
    eig_num_QR = QR_solver(H, iters=1000)
    perm = eigvals.argsort()
    perm_num_PI = eig_num_PI.argsort()
    perm_num_QR = eig_num_QR.argsort()
    eigvals_an[i] = eigvals[perm]
    eigvals_num_PI[i] = eig_num_PI[perm_num_PI]
    eigvals_num_QR[i] = eig_num_QR[perm_num_QR]

""" 
In order for us to be able to use the VQE, we'll have to rewrite 
the Hamiltonian matrix of the lipkin model first in terms of the 
Pauli operators, and then use the pauli operators to create gate 
equivalences which lead to the hamiltonian being rotated into the 
Z-basis, and thus allowing us to perform the measuremnt of the ex-
pectation value for the lowest eigenstate. We'll do it in a seperate
function, and import it to this file. The methodology used to rewrite 
the Hamiltonian matrix/our ansatz is presented in "Quantum Computing 
for Programmers" written by Robert Hundt, page 252.
"""
# Solving the eigenvalue problem using the VQE method
# Vs_vqe = np.linspace(0.0, 2.0, 10)
# learning_rate = 0.1
# momentum = 0.2
# num_shots = 2500
# max_epochs = 2500
# epochs = np.zeros(len(Vs_vqe))
# min_energy = np.zeros(len(Vs_vqe))
# for i, v in enumerate(tqdm(Vs_vqe)):
#     angles = np.random.uniform(0.0, np.pi, size=4)
#     angles, epoch, min_energy[i], delta_energy = VQE_momentum(
#         learning_rate, momentum, max_epochs, num_shots, angles, v, J=1
#     )
#     # print(f" V: {v}, Energy: {energy}, Epoch: {epoch}, Delta_energy: {delta_energy}")
#
# plot_energy(eigvals_an, min_energy, Vs_an, Vs_vqe)

# The solution above are valid for a simplified LMG model, where the term W is set to equal 0.
# Using the correspondance of the quasi-spin operators to pauli matrices, and the methodology
# presented in physics review C, we can include this term here.


Vs_an = np.linspace(0.0, 2.0, 200)
Ws_an = np.linspace(0.0, 2.0, 200)
eigvals_an_FH = np.zeros((len(Vs_an), 4))

for i, v in enumerate(tqdm(Vs_an)):
    H = lipkin_H_J1_Pauli(v, Ws_an[i], full=True)
    eigvals_FH, eigvecs_FH = np.linalg.eig(H)
    perm = eigvals_FH.argsort()
    eigvals_an_FH[i] = eigvals_FH[perm]

plot_energy_FH(eigvals_an, eigvals_an_FH, Vs_an)
