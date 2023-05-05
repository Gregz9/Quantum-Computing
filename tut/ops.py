from __future__ import annotations
import numpy as np
import math
import cmath
import tensor
import random
import state


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

    def __call__(
        self, arg: Union[state.State, ops.Operator], idx: int = 0
    ) -> state.State:
        return self.apply(arg, idx)

    def apply(self, arg: Union[state.State, ops.Operator], idx: int) -> state.State:
        """Apply an operator to a state or another operator."""

        if isinstance(arg, Operator):
            arg_bits = arg.nbits
            if idx > 0:
                arg.Identity().kpow(idx) * arg
            if self.nbits > arg.nbits:
                arg = arg * Identity().kpow(self.nbits - idx - arb_bits)

            if self.nbits != arg.nbits:
                raise AssertionError("Operator with mis-matched dimensions.")

            return arg @ self

        if not isinstance(arg, state.State):
            raise AssertionError("Invalid parameter, expected State.")

        op = self
        if idx > 0:
            op = Identity().kpow(idx) * op
        if arg.nbits - idx - self.nbits > 0:
            op = op * Identity().kpow(arg.nbits - idx - self.nbits)

        return state.State(np.matmul(op, arg))

        return state.State(np.matmul(self, arg))

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


def Identity(d: int = 1) -> Operator:
    return Operator(np.array([[1.0, 0.0], [0.0, 1.0]])).kpow(d)


def PauliX(d: int = 1) -> Operator:
    return Operator(np.array([[0.0j, 1.0], [1.0, 0.0j]])).kpow(d)


def PauliY(d: int = 1) -> Operator:
    return Operator(np.array([[0.0, -1.0j], [1.0j, 0.0]])).kpow(d)


def PauliZ(d: int = 1) -> Operator:
    return Operator(np.array([[1.0, 0.0], [0.0, -1.0]])).kpow(d)


_PAULI_X = PauliX()
_PAULI_Y = PauliY()
_PAULI_Z = PauliZ()


# TODO: FIx the code, so it takes the exact theta to be used
def Rotation(v: np.ndarray, theta: float) -> np.ndarray:
    """Produce the single-qubit rotation operator."""

    v = np.asarray(v)
    if v.shape != (3,) or not math.isclose(v @ v, 1) or not np.all(np.isreal(v)):
        raise ValueError("Rotation vector must be 3D real unit vector")

    return np.cos(theta / 2) * Identity() - 1j * np.sin(theta / 2) * (
        v[0] * _PAULI_X + v[1] * _PAULI_Y + v[2] * _PAULI_Z
    )


def RotationX(theta: float) -> Operator:
    return Rotation([1.0, 0.0, 0.0], theta)


def RotationY(theta: float) -> Operator:
    return Rotation([0.0, 1.0, 0.0], theta)


def RotationZ(theta: float) -> Operator:
    return Rotation([0.0, 0.0, 1.0], theta)


def Phase(d: int = 1) -> Operator:
    return Operator(np.array([[1.0, 0.0], [0.0, 1.0j]])).kpow(d)


def Sgate(d: int = 1) -> Operator:
    return Phase(d)


def Rk(k: int, d: int = 1) -> Operator:
    return Operator(
        np.array([(1.0, 0.0), (0.0, cmath.exp(2.0 * cmath.pi * 1j / 2**k))])
    ).kpow(d)


def U1(lam: float, d: int = 1) -> Operator:
    return Operator(np.array([(1.0, 0.0), (0.0, cmath.exp(1j * lam))])).kpow(d)


def Vgate(d: int = 1) -> Operator:
    return Operator(0.5 * np.array([(1 + 1j, 1 - 1j), (1 - 1j, 1 + 1j)])).kpow(d)


def Tgate(d: int = 1) -> Operator:
    """T-gate is sqrt(S-gate)"""

    return Operator(np.array([[1.0, 0.0], [0.0, cmath.exp(cmath.pi * 1j / 4)]])).kpow(d)


def Yroot(d: int = 1) -> Operator:
    """Root of Y-gate"""
    return Operator(0.5 * np.array([(1 + 1j, -1 - 1j), (1 + 1j, 1 + 1j)])).kpow(d)


def Hadamard(d: int = 1) -> Operator:
    return Operator(1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]])).kpow(d)
