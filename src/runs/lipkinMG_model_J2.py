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

assert np.allclose(H2, H3)

# Summing to compare to the 5x5 matrix.
# print(np.sum(H3[:, 0], axis=0))
# print(np.sum(H3[:, 1:5], axis=0))
# print(np.sum(H3[:, 5:11], axis=0))
# print(np.sum(H3[:, 11:15], axis=0))
# print(np.sum(H3[:, 15], axis=0))

v_values_an = np.linspace(0, 2.0, 100)
eigvals_an = np.zeros((len(v_values_an), 16))
entropy = np.zeros((len(v_values_an), 16))

for i, v in enumerate(tqdm(v_values_an)):
    H = lipkin_H_J2_Pauli(v)
    eig_vals, eig_vecs = np.linalg.eig(H)
    eig_perm = eig_vals.argsort()
    eigvals_an[i], eig_vecs = eig_vals[eig_perm], eig_vecs[:, eig_perm]


fig, axs = plt.subplots(1, 1, figsize=(8, 8))
for i in range(len(eigvals_an[0])):
    axs.plot(v_values_an, eigvals_an[:, i], label=f"$E_{i}$")
axs.set_xlabel(r"$V/\epsilon$")
axs.set_ylabel(r"$E/\epsilon$")
axs.legend(loc="upper left")
plt.show()
