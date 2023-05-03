from __future__ import annotations
import numpy as np
import math
import cmath
import random


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
