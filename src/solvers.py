from __future__ import annotations
import numpy as np
import math
import cmath
from typing import Iterable, List, Tuple
from src.ops import *

_PAULI_X = PauliX()
_PAULI_Y = PauliY()
_PAULI_Z = PauliZ()


def hamilitonian(dim=2, e0=0, e1=1.0, Xdiag=1.0, Xnondiag=0.0, lam=1.0):
    Hamilitonian = np.zeros((dim, dim))
    Hamilitonian[0, 0] = e0 + (lam * Xdiag)
    Hamilitonian[0, 1] = lam * Xnondiag
    Hamilitonian[1, 0] = lam * Xnondiag
    Hamilitonian[1, 1] = e1 - (lam * Xdiag)

    return Hamilitonian


def Pauli_hamiltionian(dim=2, e0=0.0, e1=0.0, Xdiag=1.0, Xnondiag=0.0, lam=1.0):
    eps = (e0 + e1) / 2
    omega = (e0 - e1) / 2
    c = (Xdiag - Xdiag) / 2
    omega_z = (Xdiag - (-Xdiag)) / 2
    omega_x = Xnondiag

    H0 = eps * Identity() + omega * _PAULI_Z
    H1 = c * Identity() + omega_z * _PAULI_Z + omega_x * _PAULI_X
    H = H0 + lam * H1
    return H


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


def power_iteration(a, num_iterations) -> (np.ndarray, np.ndarray):
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

    print("Eigenvalues:", eigenvalues[1])
    print("Eigenvectors:", eigenvectors[1])
    return eigenvalues, eigenvectors


def new_basis_1q(theta=np.pi, phi=np.pi) -> np.ndarray:
    Rx = np.cos(theta / 2) * Identity() - 1j * np.sin(theta / 2) * _PAULI_X
    Ry = np.cos(phi / 2) * Identity() - 1j * np.sin(phi / 2) * _PAULI_Y
    basis0 = np.array([1, 0])
    return Ry @ Rx @ basis0


def VQE_naive(H, inter):
    angles = (0, 180, inter)
    n = np.size(angles)
    ExpectationValues = np.zeros((n, n))
    EigValues, _ = analytical_solver(H)
    for i in range(n):
        theta = np.pi * angles[i] / 180.0
        for j in range(n):
            phi = np.pi * angles[i] / 180.0
            new_basis = new_basis_1q(theta, phi)
            Energy = new_basis.conj().T @ H @ new_basis
            Ediff = abs(np.real(EigValues[0] - Energy))
            ExpectationValues[i, j] = Ediff
    print(np.min(ExpectationValues))


if __name__ == "__main__":
    hamil = hamilitonian(dim=2, e0=0.0, e1=4.0, Xdiag=3, Xnondiag=0.2)
    hamil2 = Pauli_hamiltionian(dim=2, e0=0.0, e1=4.0, Xdiag=3.0, Xnondiag=0.2, lam=0.5)
    eig_vals, eig_vecs = analytical_solver(hamil)
    eig_vals2, eig_vecs2 = analytical_solver(hamil2)

    VQE_naive(hamil, 10)
