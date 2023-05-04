from __future__ import annotations
import numpy as np
import math
import cmath
import random
from typing import List, Iterable, Tuple, Union
import itertools


def kpow(mat, n: int) -> np.ndarray:
    if n == 0:
        return 1.0
    t = mat
    for _ in range(n - 1):
        t = np.kron(t, mat)
    return t


def nbits(psi) -> int:
    return int(np.log2(psi.shape[0]))


def bits2val(bits: List[int]) -> int:
    """For a given enumerable 'bits', compute the decimal integer."""

    return sum(v * (1 << (len(bits) - i - 1)) for i, v in enumerate(bits))


def val2bits(val: int, nbits: int) -> List[int]:
    """Convert demial integer to list of {0, 1}."""

    return [int(c) for c in format(val, "0{}b".format(nbits))]


def bitprod(nbits: int) -> Iterable[int]:
    """Produce the iterable cartesian of nbits {0, 1}."""

    for bits in itertools.product([0, 1], repeat=nbits):
        yield bits


def state_to_string(bits) -> str:
    """Convert state to string like |010>."""

    s = "".join(str(i) for i in bits)
    return "|{:s}> (|{:d}>)".format(s, int(s, 2))


def prob(psi) -> Iterable[float]:
    return [np.real(ampl.conj() * ampl) for ampl in psi]


def phase(psi) -> Iterable[float]:
    return [math.degrees(cmath.phase(ampl)) for ampl in psi]


def bitstring(*bits) -> np.ndarray:
    d = len(bits)
    if d == 0:
        raise ValueError("Rank must be at least 1.")
    t = np.zeros(1 << d, dtype=np.complex128)
    t[bits2val(bits)] = 1
    return t


def rand(n: int) -> np.ndarray:
    bits = [0] * n
    for i in range(n):
        bits[i] = random.randint(0, 1)
    return bitstring(*bits)


def density(psi) -> np.ndarray:
    return np.outer(psi, psi.conj())


def dump(psi) -> None:
    def ampl(psi, *bits) -> np.complexfloating:
        idx = bits2val(bits)
        return psi[idx]

    def prob(psi, *bits) -> float:
        amplitude = ampl(psi, *bits)
        return np.real(amplitude.conj() * amplitude)

    def phase(psi, *bits) -> float:
        amplitude = ampl(psi, *bits)
        return math.degrees(cmath.phase(amplitude))

    for bits in bitprod(nbits(psi)):
        print(state_to_string(bits))
        print(f"Amplitude: {str(ampl(psi, *bits)).strip('(').strip(')')}")
        print(f"Probability: {str(prob(psi, *bits)).strip('(').strip(')')}")
        print(f"Phase: {str(phase(psi, *bits)).strip('(').strip(')')}")


def qubit(alpha: complex = None, beta: complex = None) -> np.ndarray:
    """Creates a one-qubit basis"""

    if alpha is None and beta is None:
        raise ValueError(
            "Either alpha or beta needs to be specified in order to create a basis."
        )

    if beta is None:
        beta = np.sqrt(1.0 - np.real(np.conj(alpha) * alpha))
    elif alpha is None:
        alpha = np.sqrt(1.0 - np.real(np.conj(beta) * beta))

    if not np.isclose(
        np.real(np.conj(alpha) * alpha) + np.real(np.conj(beta) * beta), 1.0
    ):
        raise ValueError("The sum of qubit state probabilities does not sum to 1")

    qb = np.zeros(2, dtype=np.complex128)
    qb[0] = alpha
    qb[1] = beta

    return qb


def bell_state(a: int, b: int) -> np.ndarray:
    """Create one of the four bell states"""

    if a not in [0, 1] or b not in [0, 1]:
        raise ValueError("Values for a and b are required to be either 0 or 1")

    psi = bitstring(a, b)
    psi = Hadamard() @ psi
    
