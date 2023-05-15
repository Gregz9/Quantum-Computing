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
    return H, [eps, c, omega, omega_z, omega_x]


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


def ansatz_1qubit(theta, phi) -> np.ndarray:
    Rx = np.cos(theta * 0.5) * Identity() - 1j * np.sin(theta * 0.5) * _PAULI_X
    Ry = np.cos(phi * 0.5) * Identity() - 1j * np.sin(phi * 0.5) * _PAULI_Y
    basis0 = np.array([1, 0])
    return Ry @ Rx @ basis0, Rx, Ry, basis0


def measure_energy_1q(theta, phi, lmb, shots):
    _, elements = hamiltonian_1qubit(
        2, e0=0.0, e1=4.0, V11=3, V12=0.2, V21=0.2, V22=-3, lam=lmb
    )
    # elements have the following structure: [eps, c, omega, omega_z, omega_x]

    # In order to measure the estimated energy, we'll have to rewrite all
    # elements of the Hamiltonian into the pauli Z basis, as it's the only
    # of the pauli matrices with vectors |0> and |1> forming its basis.

    init_state, _, _, _ = ansatz_1qubit(theta, phi)
    _, measure_z, counts, obs_probs_z = measure(init_state, shots)

    init_state, _, _, _ = ansatz_1qubit(theta, phi)
    measure_x = Hadamard() @ init_state
    _, measure_x, counts, obs_probs_x = measure(measure_x, shots)

    exp_val_z = (
        (elements[2] + lmb * elements[3]) * (shots - 2 * np.sum(measure_z)) / shots
    )
    exp_val_x = lmb * elements[4] * (shots - 2 * np.sum(measure_x)) / shots
    exp_val_i = elements[0] + elements[1] * lmb
    exp_val = exp_val_z + exp_val_x + exp_val_i

    return exp_val


def VQE_1qubit(eta, epochs, num_shots, init_angles, lmbd):
    epoch = 0
    energy = measure_energy_1q(angles[0], angles[1], lmbd, shots)
    for epoch in range(epochs):
        grad = np.zeros((angles.shape))
        for i in range(angles.shape):
            angles_tmp = angles.copy()
            angles_tmp[i] += np.pi / 2
            ener_pl = measure_energy_1q(angles_tmp[0], angles_tmp[1], lmbd, shots)
            angles_tmp[i] -= np.pi
            ener_min = measure_energy_1q(angles_tmp[0], angles_tmp[1], lmbd, shots)
            grad[i] = (ener_pl - ener_min) / 2
        angles -= eta * grad
        new_energy = measure_energy_1q(angles[0], angles[1], lmbd, shots)
        if np.abs(new_energy - energy) < 1e-10:
            break
        energy = new_energy

    return angles, epoch, energy, delta_energy


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


def get_energy(H, theta, phi, collapse_to=0):
    basis = np.zeros(2)
    basis[collapse_to] = 1
    basis = RotationY(phi) @ RotationX(theta) @ basis
    return basis.conj().T @ H @ basis


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
