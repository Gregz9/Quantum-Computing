from __future__ import annotations
from src.solvers import *
from src.ops import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

# The case of spin J=1, solving for eigenvalues using analytical and numerical methos


def plot_energy(energy_an, energy_vqe, energy_vqe_scipy, v_values_ana, v_values_vqe):
    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    for i in range(len(energy_an[0])):
        axs.plot(v_values_ana, energy_an[:, i], label=f"$E_{i}$")
    axs.set_xlabel(r"$V/\epsilon$")
    axs.scatter(v_values_vqe, energy_vqe, label="VQE Energy", marker="o")
    axs.scatter(v_values_vqe, energy_vqe_scipy, label="VQE Energy Scipy", marker="x")
    axs.set_ylabel(r"$E/\epsilon$")
    axs.set_title("Energy levels as a function of interaction parameter V$")
    axs.legend(loc="upper left")
    plt.show()


start_time = time.time()
Vs_an = np.linspace(0.0, 2.0, 200)
eigvals_an = np.zeros((len(Vs_an), 3))

for i, v in enumerate(tqdm(Vs_an)):
    H = lipkin_H_J1(v)
    eigvals, eigvecs = np.linalg.eig(H)
    perm = eigvals.argsort()
    eigvals_an[i] = eigvals[perm]

print(f"Time taken for the analytical method: {time.time() - start_time}")

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

start_time = time.time()

# Solving the eigenvalue problem using the VQE method
Vs_vqe = np.linspace(0.0, 2.0, 10)
learning_rate = 0.1
momentum = 0.2
num_shots = 2500
max_epochs = 2500
epochs = np.zeros(len(Vs_vqe))
min_energy = np.zeros(len(Vs_vqe))
for i, v in enumerate(tqdm(Vs_vqe)):
    angles = np.random.uniform(0.0, np.pi, size=4)
    angles, epoch, min_energy[i], delta_energy = VQE_momentum(
        learning_rate, momentum, max_epochs, num_shots, angles, v, J=1
    )
print(f"Time taken for the VQE method: {time.time() - start_time}")

start_time = time.time()
min_energy_scipy = np.zeros(len(Vs_vqe))
for i, v in enumerate(tqdm(Vs_vqe)):
    angles = np.random.uniform(0.0, np.pi, size=4)
    res = minimize(
        measure_energy_J1,
        angles,
        args=(v, num_shots),
        method="Powell",
        options={"maxiter": 1000},
        tol=1e-5,
    )
    min_energy_scipy[i] = res.fun
print(f"Time taken for the VQE scipy method: {time.time() - start_time}")

plot_energy(eigvals_an, min_energy, min_energy_scipy, Vs_an, Vs_vqe)
