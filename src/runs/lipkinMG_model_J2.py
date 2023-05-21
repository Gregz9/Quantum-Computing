from src.solvers import *
from src.ops import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


v_values_an = np.linspace(0, 2.0, 100)
w_values_an = np.linspace(0, 1.0, 100)
eigvals_an = np.zeros((len(v_values_an), 16))
eigvals_full_an = np.zeros((len(v_values_an), 16))
entropy = np.zeros((len(v_values_an), 16))

for i, v in enumerate(tqdm(v_values_an)):
    H = lipkin_H_J2_Pauli(v)
    HF = lipkin_H_J2_Pauli(v, w_values_an[i], True)

    eig_vals, eig_vecs = np.linalg.eig(H)
    eig_perm = eig_vals.argsort()
    eigvals_an[i], eig_vecs = eig_vals[eig_perm], eig_vecs[:, eig_perm]

    eig_vals_full, eig_vecs_full = np.linalg.eig(HF)
    eig_perm_full = eig_vals_full.argsort()
    eigvals_full_an[i], eig_vecs_full = (
        eig_vals_full[eig_perm_full],
        eig_vecs_full[:, eig_perm_full],
    )


fig, axs = plt.subplots(1, 1, figsize=(8, 8))
for i in range(len(eigvals_full_an[0])):
    axs.plot(v_values_an, eigvals_full_an[:, i], label=f"$E_{i}$")
    axs.plot(v_values_an, eigvals_an[:, i], label=f"$E_{i}$", linestyle="dashed")
axs.set_xlabel(r"$V/\epsilon$")
axs.set_ylabel(r"$E/\epsilon$")
axs.legend(loc="upper left")
plt.show()
