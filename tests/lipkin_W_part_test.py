import numpy as np
import matplotlib.pyplot as plt
from src.ops import *
from src.solvers import *
from tqdm import tqdm

np.set_printoptions(linewidth=200)

H = lipkin_H_J2_Pauli(v=1.0, w=1.0, full=True)
v_values_an = np.linspace(0, 2.0, 100)
w_values_an = np.linspace(0, 1.0, 100)
eigvals_an = np.zeros((len(v_values_an), 16))


for i, v in enumerate(tqdm(v_values_an)):
    H = lipkin_H_J2_Pauli(v, w_values_an[i], True)

    eig_vals, eig_vecs = np.linalg.eig(H)
    eig_perm = eig_vals.argsort()
    eigvals_an[i], eig_vecs = eig_vals[eig_perm], eig_vecs[:, eig_perm]

circuits = prep_circuit_lipkin_J2()
# fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# for i in range(len(eigvals_an[0])):
#     axs.plot(v_values_an, eigvals_an[:, i], label=f"$E_{i}$")
# axs.set_xlabel(r"$V/\epsilon$")
# axs.set_ylabel(r"$E/\epsilon$")
# axs.legend(loc="upper left")
# plt.show()
