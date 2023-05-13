from __future__ import annotations
import numpy as np
import math
import cmath
import random
from src.qubit import *


def Identity(d: int = 1) -> np.ndarray:
    return kpow(np.array([[1.0, 0.0], [0.0, 1.0]]), d)


def PauliX(d: int = 1) -> np.ndarray:
    return kpow(np.array([[0.0j, 1.0], [1.0, 0.0j]]), d)


def PauliY(d: int = 1) -> np.ndarray:
    return kpow(np.array([[0.0, -1.0j], [1.0j, 0.0]]), d)


def PauliZ(d: int = 1) -> np.ndarray:
    return kpow(np.array([[1.0, 0.0], [0.0, -1.0]]), d)


_PAULI_X = PauliX()
_PAULI_Y = PauliY()
_PAULI_Z = PauliZ()


def Rotation(v: np.ndarray, theta: float) -> np.ndarray:
    v = np.asarray(v)
    if v.shape != (3,) or not math.isclose(v @ v, 1) or not np.all(np.isreal(v)):
        raise ValueError("Rotation vector must be 3D real unit vector")

    return np.cos(theta / 2) * Identity() - 1j * np.sin(theta / 2) * (
        v[0] * _PAULI_X + v[1] * _PAULI_Y + v[2] * _PAULI_Z
    )


def RotationX(theta: float) -> np.ndarray:
    return Rotation([1.0, 0.0, 0.0], theta)


def RotationY(theta: float) -> np.ndarray:
    return Rotation([0.0, 1.0, 0.0], theta)


def RotationZ(theta: float) -> np.ndarray:
    return Rotation([0.0, 0.0, 1.0], theta)


def Phase(d: int = 1) -> np.ndarray:
    return kpow(np.array([[1.0, 0.0], [0.0, 1.0j]]), d)


def Sgate(d: int = 1) -> np.ndarray:
    return Phase(d)


def Rk(k: int, d: int = 1) -> np.ndarray:
    return kpow(
        np.array([(1.0, 0.0), (0.0, cmath.exp(2.0 * cmath.pi * 1j / 2**k))]), d
    )


def U1(lam: float, d: int = 1) -> np.ndarray:
    return kpow(np.array([(1.0, 0.0), (0.0, cmath.exp(1j * lam))]), d)


def Vgate(d: int = 1) -> np.ndarray:
    return kpow((0.5 * np.array([(1 + 1j, 1 - 1j), (1 - 1j, 1 + 1j)])), d)


def Tgate(d: int = 1) -> np.ndarray:
    return kpow(np.array([[1.0, 0.0], [0.0, cmath.exp(cmath.pi * 1j / 4)]]), d)


def Yroot(d: int = 1) -> np.ndarray:
    return kpow(0.5 * np.array([(1 + 1j, -1 - 1j), (1 + 1j, 1 + 1j)]), d)


def Hadamard(d: int = 1) -> np.ndarray:
    return kpow((1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]])), d)


def Projector(psi: np.ndarray) -> np.ndarray:
    """Construct a projection operator for basis state."""

    return density(psi)


def ControlledU(idx0: int, idx1: int, oper: np.ndarray) -> np.ndarray:
    if idx0 == idx1:
        raise ValueError("Control and controlled qubit must not be equal.")

    p0 = Projector(zeros())
    p1 = Projector(ones())
    ifill = Identity(abs(idx1 - idx0) - 1)
    ufill = kpow(Identity(), nbits(oper))

    if idx1 > idx0:
        if idx1 - idx0 > 1:
            op = np.kron(np.kron(p0, ifill), ufill) + np.kron(np.kron(p1, ifill), oper)
        else:
            op = np.kron(p0, ufill) + np.kron(p1, oper)
    else:
        if idx0 - idx1 > 1:
            op = np.kron(np.kron(ufill, ifill), p0) + np.kron(np.kron(oper, ifill), p1)
        else:
            op = np.kron(ufill, p0) + np.kron(oper, p1)
    return op


def Cnot(idx0: int = 0, idx1: int = 1) -> np.ndarray:
    return ControlledU(idx0, idx1, PauliX())


def measurement(
    psi, operator: np.ndarray, shots: int = 1
) -> (np.ndarray, np.ndarray, np.float64):
    psi = operator @ psi
    probs = np.around(np.diag(density(psi)).real, 3)
    possible_states = np.arange(len(psi))
    measurement = np.random.choice(possible_states, p=probs, size=shots)

    collapsed_psi = np.zeros(len(psi))
    collapsed_psi[measurement[-1]] = 1
    counts = np.unique(measurement, return_counts=True)[1]
    return probs, counts, np.around(counts / shots, 3), psi


def measure(psi: np.ndarray, shots) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    probs = np.abs(psi) ** 2
    possible = np.arange(len(psi))
    outcome = np.random.choice(possible, p=probs, size=shots)

    psi = np.zeros(len(psi))
    psi[outcome[-1]] = 1
    counts = np.unique(outcome, return_counts=True)[1]
    return psi, outcome, counts, counts / shots

    # if collapse:
    #     mvmul = np.dot(op, psi)
    #     divisor = np.real(np.linalg.norm(mvmul))
    #     if divisor > 1e-10:
    #         normed = mvmul / divisor
    #     else:
    #         raise AssertionError("Measure() collapses to 0.0 probability state.")
    #
    #     return np.real(prob0), normed
    #
    # return np.real(prob0), psi
