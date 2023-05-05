import state
import random
import numpy as np
import scipy

if __name__ == "__main__":
    # psi = state.qubit(alpha=1.0)
    # phi = state.qubit(beta=1.0)
    #
    # combo = psi * phi
    # print(combo)

    p1 = state.qubit(alpha=np.random.random())
    x1 = state.qubit(alpha=np.random.random())

    # psi = p1 * x1  # Kronecker product (Tensor product)

    # print(psi)
    # inner product of full qubit states
    # assert np.allclose(np.inner(psi.conj(), psi), 1.0)

    # assert np.allclose(np.inner(p1.conj(), p1) * np.inner(x1.conj(), x1), 1.0)

    # print(psi.ampl(1, 1))
    # print(psi.prob(1, 1))
    # print(psi.phase(1, 0))
    # print(psi.dump())

    # psi = state.rand(1)

    print(state.zeros(1))
    print(state.zeros(1).density())
