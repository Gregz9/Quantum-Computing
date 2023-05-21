from src.solvers import *
from src.ops import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)
I = Identity()
X = PauliX()
Y = PauliY()
Z = PauliZ()
ZIII = np.kron(Z, np.kron(I, np.kron(I, I)))
IZII = np.kron(I, np.kron(Z, np.kron(I, I)))
IIZI = np.kron(I, np.kron(I, np.kron(Z, I)))
IIIZ = np.kron(I, np.kron(I, np.kron(I, Z)))

XXII = np.kron(X, np.kron(X, np.kron(I, I)))
XIXI = np.kron(X, np.kron(I, np.kron(X, I)))
XIIX = np.kron(X, np.kron(I, np.kron(I, X)))
IXXI = np.kron(I, np.kron(X, np.kron(X, I)))
IXIX = np.kron(I, np.kron(X, np.kron(I, X)))
IIXX = np.kron(I, np.kron(I, np.kron(X, X)))

YYII = np.kron(Y, np.kron(Y, np.kron(I, I)))
YIYI = np.kron(Y, np.kron(I, np.kron(Y, I)))
YIIY = np.kron(Y, np.kron(I, np.kron(I, Y)))
IYYI = np.kron(I, np.kron(Y, np.kron(Y, I)))
IYIY = np.kron(I, np.kron(Y, np.kron(I, Y)))
IIYY = np.kron(I, np.kron(I, np.kron(Y, Y)))
H = lipkin_H_J2_Pauli(v=1.0, w=2.0)
w = 1.0
H2_2 = -(w / 2) * (
    (XXII + XIXI + XIIX + IXXI + IXIX + IIXX + YYII + YIYI + YIIY + IYYI + IYIY + IIYY)
)

Jm = (X - 1j * Y) / np.sqrt(2)
Jp = (X + 1j * Y) / np.sqrt(2)

JpJm = np.kron(Jp, Jm)
JmJp = np.kron(Jm, Jp)

JpJmII = np.kron(Jp, np.kron(Jm, np.kron(I, I)))
JmJpII = np.kron(Jm, np.kron(Jp, np.kron(I, I)))
JpIJmI = np.kron(Jp, np.kron(I, np.kron(Jm, I)))
JmIJpI = np.kron(Jm, np.kron(I, np.kron(Jp, I)))
JpIIJm = np.kron(Jp, np.kron(I, np.kron(I, Jm)))
JmIIJp = np.kron(Jm, np.kron(I, np.kron(I, Jp)))
IJpJmI = np.kron(I, np.kron(Jp, np.kron(Jm, I)))
IJmJpI = np.kron(I, np.kron(Jm, np.kron(Jp, I)))
IJpIJm = np.kron(I, np.kron(Jp, np.kron(I, Jm)))
IJmIJp = np.kron(I, np.kron(Jm, np.kron(I, Jp)))
IIJpJm = np.kron(I, np.kron(I, np.kron(Jp, Jm)))
IIJmJp = np.kron(I, np.kron(I, np.kron(Jm, Jp)))

N = 4
w = 1.0
H2 = -(w / 2) * (
    JpJmII
    + JmJpII
    + JpIJmI
    + JmIJpI
    + JpIIJm
    + JmIIJp
    + IJpJmI
    + IJmJpI
    + IJpIJm
    + IJmIJp
    + IIJpJm
    + IIJmJp
)

H3 = -(w / 2) * (
    (XXII + XIXI + XIIX + IXXI + IXIX + IIXX + YYII + YIYI + YIIY + IYYI + IYIY + IIYY)
)
# H2_red = np.zeros((16, 5))
# H2_red[:, 0] = H2[:, 0]
# H2_red[:, 1] = np.sum(H2[:, 1:5], axis=1)
# H2_red[:, 2] = np.sum(H2[:, 5:11], axis=1)
# H2_red[:, 3] = np.sum(H2[:, 11:15], axis=1)
# H2_red[:, 4] = H2[:, 15]
# H_red_red = np.zeros((5, 5))
# H_red_red[0, :] = H2_red[0, :]
# H_red_red[1, :] = np.sum(H2_red[1:5, :], axis=0)
# H_red_red[2, :] = np.sum(H2_red[5:11, :], axis=0)
# H_red_red[3, :] = np.sum(H2_red[11:15, :], axis=0)
# H_red_red[4, :] = np.sum(H2_red[15, :], axis=0)


# assert np.allclose(H2, H3)
#
# Summing to compare to the 5x5 matrix.
# print(np.sum(H3[:, 0], axis=0))
# print(np.sum(H3[:, 1:5], axis=0))
# print(np.sum(H3[:, 5:11], axis=0))
# print(np.sum(H3[:, 11:15], axis=0))
# print(np.sum(H3[:, 15], axis=0))

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
