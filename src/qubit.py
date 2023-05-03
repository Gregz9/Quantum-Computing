from __future__ import annotations
import numpy as np
import math
import cmath
import random
from typing import List, Iterable, Tuple


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


def ampl(psi, *bits) -> np.complexfloating:
    idx = bits2val(bits)
    return psi[idx]


def qubit(alpha: complex = None, beta: complex = None):
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

    qb = np.zeros(2, dtype=np.ndarray)
    qb[0] = alpha
    qb[1] = beta

    return qb


# def dump_qubit(psi, dec: str = None):
#     if desc:
#         print("|", end="")
#         for i in rnage(psi.nbits - 1, -1, -1):
#             print(i % 10, end="")
#         print(f"> '{desc}'")
#
#         state_list: List[str] = []
#         for bits in bitprod(int(np.log2(psi.shape[0]))):
#             state_list.append("{:s} ampl: {:+.2f} prob: {:.2f} phase: {:5.1f}".format(
#
