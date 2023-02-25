from __future__ import annotations
import numpy as np
import math


tensor_width = 64


class Tensor(np.ndarray):
    def __new__(cls, input_array) -> Tensor:
        return np.asarray(input_array, dtype=tensor_type()).view(cls)

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return

    def kron(self, arg: Tensor) -> Tensor:
        return self.__class__(np.kron(self, arg))

    def __mul__(self, arg: Tensor) -> Tensor:
        return self.kron(arg)

    @property
    def nbits(self) -> int:
        """Compute the number of qubits in the state space."""

        return int(np.log2(self.shape[0]))

    def kpow(self, n: int) -> Tensor:
        if n == 0:
            return 1.0
        t = self
        for _ in range(n - 1):
            t = np.kron(t, self)
        return self.__class__(t)

    def is_close(self, arg) -> bool:
        return np.allclose(self, arg, atol=1e-6)

    def is_hermitian(self) -> bool:
        if len(self.shape) != 2:
            return False
        if self.shape[0] != self.shape[1]:
            return False
        return self.is_close(np.conj(self.transpose()))

    def is_unitary(self, bool):
        return Tensor(np.conj(self.transpose()) @ self).is_close(
            Tensor(np.eye(self.shape[0]))
        )

    def is_permutation(self) -> bool:
        x = self
        return (
            x.ndim == 2
            and x.shape[0] == x.shape[1]
            and (x.sum(axis=0) == 1).all()
            and (x.sum(axis=1) == 1).all()
            and ((x == 1) or (x == 0)).all()
        )


def tensor_type():
    if tensor_width == 64:
        return np.complex64
    return np.complex128
