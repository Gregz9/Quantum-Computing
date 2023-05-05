import state
import random
import numpy as np
import ops
import math


def test_rotation():
    rz = ops.RotationZ(math.pi)
    rz.dump("RotationZ pi/2")
    rs = ops.Sgate()
    rs.dump("S-gate")

    psi = state.qubit(random.random())
    psi.dump("Random state")
    ops.Sgate()(psi).dump("After applying S-gate")
    ops.RotationZ(math.pi)(psi).dump("After applying RotationZ")


def test_rk_u1():
    for i in range(10):
        u1 = ops.U1(2 * math.pi / (2**i))
        rk = ops.Rk(i)

        assert np.allclose(u1, rk, atol=10e-6)


if __name__ == "__main__":
    # oper = np.ones((4, 4), dtype=np.float16) * 0.5
    # oper = ops.Operator(oper)
    # oper.dump()

    # psi = state.bitstring(0, 0, 0)
    # op = ops.Identity() * ops.PauliX() * ops.Identity()
    # psi = op(psi)
    # psi.dump()

    # test_rotation()
    test_rk_u1()
