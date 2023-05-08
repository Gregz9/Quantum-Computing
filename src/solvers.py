from __future__ import annotations
import numpy as np
import math
import cmath
from typing import Iterable, List, Tuple
from src.ops import *

_PAULI_X = PauliX()
_PAULI_Y = PauliY()
_PAULI_Z = PauliZ()
_HADAMARD = Hadamard()
_SGATE = Sgate()


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


def new_basis_1q(theta, phi) -> np.ndarray:
    Rx = np.cos(theta * 0.5) * Identity() - 1j * np.sin(theta * 0.5) * _PAULI_X
    Ry = np.cos(phi * 0.5) * Identity() - 1j * np.sin(phi * 0.5) * _PAULI_Y
    basis0 = np.array([0, 1])
    return Ry @ Rx @ basis0, Rx, Ry, basis0


def new_basis_2q(theta, phi, idx=0) -> np.ndarray:
    Rx1 = np.cos(theta * 0.5) * Identity() - 1j * np.sin(theta * 0.5) * _PAULI_X
    Rx2 = np.cos(phi * 0.5) * Identity() - 1j * np.sin(phi * 0.5) * _PAULI_X
    Ry1 = np.cos(theta * 0.5) * Identity() - 1j * np.sin(theta * 0.5) * _PAULI_Y
    Ry2 = np.cos(phi * 0.5) * Identity() - 1j * np.sin(phi * 0.5) * _PAULI_Y

    Rz1 = np.cos(theta * 0.5) * Identity() - 1j * np.sin(theta * 0.5) * _PAULI_Z
    Rz2 = np.cos(phi * 0.5) * Identity() - 1j * np.sin(phi * 0.5) * _PAULI_Z
    CX_01 = Cnot(0, 1)
    basis = np.zeros(4)
    basis[idx] = 1
    return (
        # np.kron(Ry1, Ry2) @ np.kron(Rz1, Rz2) @ Cnot(0, 1) @ basis,
        np.kron(Rz1, Rz2) @ Cnot(0, 1) @ np.kron(Rx1, Rx2) @ basis,
        Ry1,
        Ry2,
        basis,
    )


def VQE_naive(H, inter):
    angles = np.arange(0, 180, inter)
    n = np.size(angles)
    ExpectationValues = np.zeros((n, n))
    Energies = np.zeros((n, n))
    EigValues, _ = analytical_solver(H)
    for i in range(n):
        theta = np.pi * angles[i] / 180.0
        for j in range(n):
            phi = np.pi * angles[j] / 180.0
            new_basis, _, _, _ = new_basis_1q(theta, phi)
            Energy = new_basis.conj().T @ H @ new_basis
            Ediff = abs(np.real(EigValues[0] - Energy))
            ExpectationValues[i, j] = Ediff
            Energies[i, j] = Energy

    # print(np.min(ExpectationValues))
    ind = np.argmin(ExpectationValues)
    # print(ExpectationValues.flatten()[5])
    print(Energies.flatten()[ind])


def VQE_naive_2q(H, inter):
    angles = np.arange(0, 180, inter)
    n = np.size(angles)
    ExpectationValues = np.zeros((n, n))
    Energies = np.zeros((n, n))
    EigValues, _ = analytical_solver(H)
    for i in range(n):
        theta = np.pi * angles[i] / 180.0
        for j in range(n):
            phi = np.pi * angles[j] / 180.0
            new_basis, Rx, Ry, basis = new_basis_2q(theta, phi)
            Energy = new_basis.conj().T @ H @ new_basis
            Ediff = abs(np.real(EigValues[0] - Energy))
            ExpectationValues[i, j] = Ediff
            Energies[i, j] = Energy

    print(np.min(np.real(ExpectationValues)))
    ind = np.argmin(ExpectationValues)
    # print(ExpectationValues.flatten()[5])
    print(Energies.flatten()[ind])


if __name__ == "__main__":
    # hamil = hamilitonian(dim=2, e0=0.0, e1=4.0, Xdiag=3, Xnondiag=0.2)
    # hamil2 = Pauli_hamiltionian(dim=2, e0=0.0, e1=4.0, Xdiag=3.0, Xnondiag=0.2, lam=0.5)
    # eig_vals, eig_vecs = analytical_solver(hamil)
    # print(eig_vals)
    # eig_vals2, eig_vecs2 = analytical_solver(hamil2)
    # VQE_naive(hamil, 10)

    # mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    # eig_vals, eig_vecs = analytical_solver(mat)
    # print(eig_vals)

    new_basis, Rx, Ry, basis = new_basis_1q(np.pi / 2, np.pi / 2)

    e00, e10, e01, e11 = 0.0, 2.5, 6.5, 7.0
    Hx, Hz = 2.0, 3.0
    mat = np.zeros((4, 4))
    mat[0, 0], mat[1, 1], mat[2, 2], mat[3, 3] = e00 + Hz, e10 - Hz, e01 - Hz, e11 + Hz
    mat[0, 3], mat[1, 2], mat[2, 1], mat[3, 0] = Hx, Hx, Hx, Hx

    eig_vals, eig_vecs = analytical_solver(mat)

    # new_basis, Rx, Ry, basis0 = new_basis_1q(np.pi / 2, np.pi / 2)
    # print(Rx)
    # print(Ry)
    # print(Ry @ Rx)
    # print(np.kron(Ry @ Rx, Identity()))
    # print(np.kron(Identity(), Ry @ Rx))
