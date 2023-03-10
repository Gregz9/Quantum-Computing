import state
import random
import numpy as np
import ops

if __name__ == "__main__":
    # oper = np.ones((4, 4), dtype=np.float16) * 0.5
    # oper = ops.Operator(oper)
    # oper.dump()

    psi = state.bitstring(0, 0, 0)
    op = ops.Identity() * ops.PauliX() * ops.Identity()
    psi = op(psi)
    psi.dump()
