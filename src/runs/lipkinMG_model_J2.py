from src.solvers import *
from src.ops import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# solving for the ground state energy of the lipkin Hamiltonian of the system with total spin J=2,
# meaning that the system conatains four fermions which are spread across two energy levels.

v_values_an = np.linspace(0, 2.0, 100)
eigvals_an = np.zeros((len(v_values_an), 16))

start_time = time.time()
for i, v in enumerate(tqdm(v_values_an)):
    H = lipkin_H_J2_Pauli(v)  # ,w, full=True)
    eig_vals, eig_vecs = np.linalg.eig(H)
    eig_perm = eig_vals.argsort()
    eigvals_an[i], eig_vecs = eig_vals[eig_perm], eig_vecs[:, eig_perm]
print(f"Time taken for the analytical method: {time.time() - start_time}")

# In order to be able to perform the measurement in the Z basis, we will need to
# rotate our measurement basis into the ZIII basis. In order to do so, we are going
# to use the approach presented by Robert Hundt in his book on Quantum computing.

# Proceeding from here, we shall prepare an 4 qubit ansatz and use it with our VQE
# algorithm to solve for the lowest eigenvalue of the the lipkin system with a spin og
# J=2, indicating four fermions.


start_time = time.time()
unitaries = prep_circuit_lipkin_J2()
v_vals = np.linspace(0.0, 2.0, 8)
w_vals = np.linspace(0.0, 2.0, 8)
ener = VQE_scipy(
    measure_energy_mul,
    v_vals,
    5 * 8,
    1000,
    0.0,
    2 * np.pi,
    "BFGS",
    # w_vals,
    # full=True,
)
print(f"Time taken for the VQE method: {time.time() - start_time}")

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
for i in range(len(eigvals_an[0])):
    axs.plot(v_values_an, eigvals_an[:, i], label=f"$E_{i}$")
axs.scatter(v_vals, ener, label="VQE Energy Scipy", marker="o")
axs.set_xlabel(r"$V/\epsilon$")
axs.set_ylabel(r"$E/\epsilon$")
axs.set_title("Energy levels as a function of interaction parameter W")
axs.legend(loc="upper left")
plt.show()
