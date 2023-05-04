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
