from src.solvers import *
from src.ops import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


v_values_an = np.linspace(0, 2.0, 100)
w_values_an = np.linspace(0, 1.0, 100)
eigvals_an = np.zeros((len(v_values_an), 16))
entropy = np.zeros((len(v_values_an), 16))

for i, v in enumerate(tqdm(v_values_an)):
    H = lipkin_H_J2_Pauli(v)
    HF = lipkin_H_J2_Pauli(v, w_values_an[i], True)

    eig_vals, eig_vecs = np.linalg.eig(H)
    eig_perm = eig_vals.argsort()
    eigvals_an[i], eig_vecs = eig_vals[eig_perm], eig_vecs[:, eig_perm]

print(eigvals_an[49][0])


# fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# for i in range(len(eigvals_an[0])):
#     axs.plot(v_values_an, eigvals_an[:, i], label=f"$E_{i}$", linestyle="dashed")
# axs.set_xlabel(r"$V/\epsilon$")
# axs.set_ylabel(r"$E/\epsilon$")
# axs.legend(loc="upper left")
# plt.show()

# In order to be able to perform the measurement in the Z basis, we will need to
# rotate our measurement basis into the ZIII basis. In order to do so, we are going
# to use the approach presented by Robert Hundt in his book on Quantum computing.

# Proceeding from here, we shall prepare an 4 qubit ansatz and use it with our VQE
# algorithm to solve for the lowest eigenvalue of the the lipkin system with a spin og
# J=2, indicating four fermions.


def get_gradient(angles, v, number_shots, unitaries):
    grad = np.zeros(len(angles))
    for index, angle in enumerate(angles):
        tmp = angles.copy()
        tmp[index] += np.pi / 2
        energy_plus = measure_energy_mul(tmp, v, number_shots)
        tmp[index] -= np.pi
        energy_minus = measure_energy_mul(tmp, v, number_shots)
        grad[index] = (energy_plus - energy_minus) / 2
    return grad


unitaries = prep_circuit_lipkin_J2()
from scipy.optimize import minimize

# number_shots = 1000
v_vals = np.array([1.0])  # only 2.0
# # target = eigvecs[:, 0] # ground state
# min_energy_scipy = np.zeros(len(v_vals))
# for index, v in enumerate(tqdm(v_vals)):
#     angles_start = np.random.uniform(low=0, high=2 * np.pi, size=5 * 8)
#     res = minimize(
#         fun=measure_energy_mul,
#         jac=get_gradient,
#         x0=angles_start,
#         args=(v, number_shots, unitaries),
#         method="BFGS",
#         options={"maxiter": 10000},
#         tol=1e-11,
#     )
#     min_energy_scipy[index] = res.fun
#     print(res.fun)


ener = VQE_scipy(
    measure_energy_mul, v_vals, 5 * 8, 1000, unitaries, 0.0, 2 * np.pi, "BFGS"
)

print(ener)
