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


def RotationX(theta: float) -> np.ndarray:
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

    p0 = Projector(np.zeros(1, dtype=np.complex64))
    p1 = Projector(np.ones(1, dtype=np.complex64))
    ifill = Identity(abs(idx1 - idx0) - 1)
    ufill = kpow(Identity(), nbits(oper))

    # print(p0)
    print(p1)

    # if idx1 > idx0:
    #     if idx1 - idx0 > 1:
    #         op = p0
