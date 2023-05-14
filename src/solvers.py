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


def hamiltonian_1qubit(
    dim=2, e0=0.0, e1=0.0, V11=1.0, V12=1.0, V21=1.0, V22=1.0, lam=1.0
):
    eps = (e0 + e1) / 2
    omega = (e0 - e1) / 2
    c = (V11 + V22) / 2
    omega_z = (V11 - V22) / 2
    omega_x = V21

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

    print(np.min(ExpectationValues))
    ind = np.argmin(ExpectationValues)
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
    print(Energies.flatten()[ind])


def get_energy(H, theta, phi, collapse_to=0):
    basis = np.zeros(2)
    basis[collapse_to] = 1
    basis = RotationY(phi) @ RotationX(theta) @ basis
    return basis.conj().T @ H @ basis

    return proj_oper.conj().T @ H @ proj_oper


def VQE_1q(H, epochs=100, eta=0.1):
    theta = 2 * np.pi * np.random.rand()
    phi = 2 * np.pi * np.random.rand()
    pi2 = 0.5 * np.pi
    for epoch in range(epochs):
        theta_grad = 0.5 * (
            get_energy(H, theta + pi2, phi) - get_energy(H, theta - pi2, phi)
        )
        phi_grad = 0.5 * (
            get_energy(H, theta, phi + pi2) - get_energy(H, theta - pi2, phi)
        )
        theta -= eta * theta_grad
        phi -= eta * phi_grad
    print(abs(get_energy(H, theta, phi)))
