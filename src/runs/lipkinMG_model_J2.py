from src.solvers import *
from src.ops import *
import numpy as np
import tqdm
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
# w = 1.0
# H2 = -(w / 2) * (XXII + XIXI + XIIX + IXXI + IXIX + IIXX) - (w / 2) * (
#     YYII + YIYI + YIIY + IYYI + IYIY + IIYY
# )

Jm = (X - 1j * Y) / np.sqrt(2)
Jp = (X + 1j * Y) / np.sqrt(2)

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

w = 1.0
H2 = (w / 2) * (
    -4
    + JpJmII
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

H_sum14 = np.sum(H[:, 1:4], axis=1)
print(np.sum(H_sum14[1:4]))
