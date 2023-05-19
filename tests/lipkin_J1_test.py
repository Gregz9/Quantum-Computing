from src.solvers import *
from src.ops import *
import numpy as np

I = Identity()
X = PauliX()
Y = PauliY()
Z = PauliZ()
XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZI = np.kron(Z, I)
IZ = np.kron(I, Z)
XY = np.kron(X, Y)
YX = np.kron(Y, X)
W = np.random.uniform(0.0, 2.0)
V = np.random.uniform(0.0, 2.0)

H = (1 / 2) * (ZI + IZ) - (V / 2) * (XX - YY) - (W / 2) * (XX + YY)

eigvals, eigvecs = np.linalg.eig(H)
print(eigvals)
