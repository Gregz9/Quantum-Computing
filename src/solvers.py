from __future__ import annotations
import numpy as np
import math
import cmath
from typing import Iterable, List, Tuple


def hamilitonian(dim=2, e0=0, e1=1.0, Xdiag=1.0, Xnondiag=0.0):
    Hamilitonian = np.zeros((dim, dim))
    Hamilitonian[0, 0] = e0 + Xdiag
    Hamilitonian[0, 1] = Xnondiag
    Hamilitonian[1, 0] = Xnondiag
    Hamilitonian[1, 1] = e1 - Xdiag

    return Hamilitonian


def analytical_solver(mat):
    EigValues, EigVectors = np.linalg.eig(mat)
    permute = EigValues.argsort()
    Eigvalues, EigVectors = EigValues[permute], EigVectors[permute]

    return Eigvalues, EigVectors


if __name__ == "__main__":
    hamil = hamilitonian(dim=2, e0=0.0, e1=4.0, Xdiag=3, Xnondiag=0.2)
    eig_vals, eig_vecs = analytical_solver(hamil)
    print(eig_vals)
    print(eig_vecs)
