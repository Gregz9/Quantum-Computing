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


def normalize(x):
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n


def power_iter(mat, iters=100):
    x = [1, 1]
    y = [1, 1]
    for i in range(iters):
        x = np.dot(mat, x)
        lambda_1, x = normalize(x)

    print("Eigenvalue:", lambda_1)
    print("Eigenvector:", x)


def power_iteration(a, num_iterations):
    n = len(a)
    eigenvalues = []
    eigenvectors = []

    for i in range(n):
        x = np.random.rand(n)

        for j in range(num_iterations):
            x = np.dot(a, x)
            eigenvalue, x = normalize(x)

        eigenvectors.append(x)
        eigenvalues.append(eigenvalue)
        a = a - eigenvalue * np.outer(x, x)

    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:", eigenvectors)
    return eigenvalues, eigenvectors


if __name__ == "__main__":
    hamil = hamilitonian(dim=2, e0=0.0, e1=4.0, Xdiag=3, Xnondiag=0.2)
    eig_vals, eig_vecs = analytical_solver(hamil)
    print(eig_vals)
    print(eig_vecs)
    power_iter(hamil)
    power_iteration(hamil, 100)
