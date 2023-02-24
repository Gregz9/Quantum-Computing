from __future__ import annotations
import numpy as np
import math
import cmath
import tensor
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

    def phase(self, *bits) -> float:
        """Compute the phase of a state from the complex amplitude."""

        amplitude = self.ampl(*bits)
        return math.degrees(cmath.phase(amplitude))

    def state_to_string(bits) -> str:
        """Convert state to string like |010>."""

        s = "".join(str(i) for i in bits)
        return "|{:s}> (|{:d}>)".format(s, int(s, 2))

    def dump_state(psi, desc: Optional[str] = None, prob_only: bool = True) -> None:
        """Dump probabilities for a state, as well as local qubit state."""

        if desc:
            print("/", end="")
            for i in range(psi.nbits - 1, -1, -1):
                print(i % 10, end="")
            print(f"> '{desc}'")

        state_list: List[str] = []
        for bits in helper.bitprod(psi.nbits): 
            if prob_only and (psi.prob(*bits) < 10e-6): 
                continue
            state_list.append(



def qubit(
    alpha: Optional[np.complexfloating] = None,
    beta: Optional[np.complexfloating] = None,
) -> State:
    if alpha is None and beta is None:
        raise ValueError("alpha, or beta, or both are required")

    if beta is None:
        beta = np.sqrt(1.0 - np.conj(alpha) * alpha)

    if alpha is None:
        alpha = np.sqrt(1.0 - np.conj(beta) * beta)

    if not math.isclose(np.conj(alpha) * alpha + np.conj(beta) * beta, 1.0):
        raise ValueError("Qubit probabilities do not add up to 1")

    qb = np.zeros(2, dtype=tensor.tensor_type())
    qb[0] = alpha
    qb[1] = beta
    return State(qb)
