from __future__ import annotations
import numpy as np
import math
import cmath
import tensor
import random


class Operator(tensor.Tensor):
    """Operators are represented by square, unitary matrices."""

    def __repr__(self) -> str:
        s = "Opertor("
        s += "Tensor:\n"
        s += super().__str__()
        return s

    def __str__(self) -> str:
        s = f"Operator for {self.nbits}-qubit state space."
        s += " Tensor:\n"
        s += super().__str__()
        return s

    def adjoint(self) -> Operator:
        return Operator(np.conj(self.T))

    def dump(self, description: Optional[str] = None, zeros: bool = False) -> None:
        res = ""
        if description:
            res += f"{description} ({self.nbits}-qubits operator)\n"
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                val = self[row, col]
                res += f"{val.real:+.1f}{val.imag:+.1f}j "
            res += "\n"
        if not zeros:
            res = res.replace("+0.0j", "    ")
            res = res.replace("+0.0", " -  ")
            res = res.replace("-0.0", " -  ")
            res = res.replace("+", " ")
        print(res)
