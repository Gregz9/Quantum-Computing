from __future__ import annotations
import numpy as np
import math
import cmath
from tut import tensor
import random

from lib import helper
from typing import List, Tuple, Iterable, Optional


class State(tensor.Tensor):
    """class State represents single and multiple-qubit states."""

    def __repr__(self) -> str:
        s = "State("
        s += super().__str__().replace("\n", "\n" + " " * len(s))
        s += ")"

        return s

    def __str__(self) -> str:
        s = f"{self.nbits}-qubit state."
        s += " Tensor:\n"
        s += super().__str__()

        return s

    def ampl(self, *bits) -> np.complexfloating:
        """Return amplitude for state indexed by 'bits'."""

        idx = helper.bits2val(bits)
        return self[idx]

    def prob(self, *bits) -> float:
        """Return the probability for the state indexed by 'bits'."""

        amplitude = self.ampl(*bits)
        return np.real(amplitude.conj() * amplitude)

    def max_prob(self) -> (List[float], float):
        """Find the state with the highest probability."""

        maxbits, max_prob = [], 0.0
        for bits in helper.bitprod(self.nbits):
            cur_prob = self.prob(*bits)
            if cur_prob > max_prob:
                maxbits, max_prob = bits, cur_prob
        return maxbits, maxprob

    def normalize(self) -> None:
        """Renormalize the state. Sum of norms==1.0."""

        dprod = np.conj(self) @ self

        if dprod.is_close(0.0):
            raise AssertionError("Normalizing to zero-probability state.")

        self /= np.sqrt(np.real(dprod))

    def phase(self, *bits: Tuple[int]) -> float:
        """Return phase of a state from the complex amplitude."""

        amplitude = self.ampl(*bits)
        return math.degrees(cmath.phase(amplitude))

    def dump(self, desc: Optional[str] = None, prob_only: bool = True) -> None:
        dump_state(self, desc, prob_only)

    def density(self) -> tensor.Tensor:
        return tensor.Tensor(np.outer(self, self.conj()))


def state_to_string(bits) -> str:
    """Convert state to string like |010>."""

    s = "".join(str(i) for i in bits)
    return "|{:s}> (|{:d}>)".format(s, int(s, 2))


def dump_state(psi, desc: str = None, prob_only: bool = True) -> None:
    """Dump probabilities for a state, as well as local qubit state."""

    if desc:
        print("|", end="")
        for i in range(psi.nbits - 1, -1, -1):
            print(i % 10, end="")
        print(f"> '{desc}'")

    state_list: List[str] = []
    for bits in helper.bitprod(psi.nbits):
        if prob_only and (psi.prob(*bits) < 10e-6):
            continue

        state_list.append(
            "{:s}:  ampl: {:+.2f} prob: {:.2f} Phase: {:5.1f}".format(
                state_to_string(bits),
                psi.ampl(*bits),
                psi.prob(*bits),
                psi.phase(*bits),
            )
        )
    state_list.sort()
    print(*state_list, sep="\n")


def qubit(alpha: complex = None, beta: complex = None) -> State:
    """Produce a given state for a single qubit."""

    if alpha is None and beta is None:
        raise ValueError("Both alpha and beta need to be specified")

    # Note tha multiplying a complex conjugate with its non-conjugate
    # is a real number, but we still have to type-cast it to avoid
    # Python warnings (hence the use of np.real()).
    if beta is None:
        beta = np.sqrt(1.0 - np.real(np.conj(alpha) * alpha))
    if alpha is None:
        alpha = np.sqrt(1.0 - np.real(np.conj(beta) * beta))

    if not math.isclose(
        np.real(np.conj(alpha) * alpha) + np.real(np.conj(beta) * beta), 1.0
    ):
        raise ValueError("Qubit probabilities do not sum to 1.")

    qb = np.zeros(2, dtype=tensor.tensor_type())
    qb[0] = alpha
    qb[1] = beta
    return State(qb)


def zeros_or_ones(d: int = 1, idx: int = 0) -> State:
    """Produce the all-0/1 basis vector for 'd' qubits"""

    if d < 1:
        raise ValueError("Rank must be at least 1")
    shape = 2**d
    t = np.zeros(shape, dtype=tensor.tensor_type())
    t[idx] = 1
    return State(t)


def zeros(d: int = 1) -> State:
    """Produce state with 'd' |0>, eg., |0000>."""
    return zeros_or_ones(d, 0)


def ones(d: int = 1) -> State:
    """Produce state with 'd' |1>, eg., |1111>."""
    return zeros_or_ones(d, 2**d - 1)


def bitstring(*bits) -> State:
    """Produce a statet from a given bit sequence, eg., |0101>."""

    d = len(bits)
    if d == 0:
        raise ValueError("Rank must be at least 1.")
    t = np.zeros(1 << d, dtype=tensor.tensor_type())
    t[helper.bits2val(bits)] = 1
    return State(t)


def rand(n: int) -> State:
    """Produce random combination of |0> and |1>."""

    bits = [0] * n
    for i in range(n):
        bits[i] = random.randint(0, 1)
    return bitstring(*bits)
