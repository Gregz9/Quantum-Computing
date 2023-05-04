from __future__ import annotations
import numpy as np
import math
import cmath
import random
from typing import List, Iterable, Tuple, Union
from src.qubit import *
from src.ops import *


def bell_state(a: int, b: int) -> np.ndarray:
    """Create one of the four bell states"""

    if a not in [0, 1] or b not in [0, 1]:
        raise ValueError("Values for a and b are required to be either 0 or 1")

    psi = bitstring(a, b)
    print(psi)
    ext_HAD = np.kron(Hadamard(), Identity())
    print(ext_HAD)
    psi = ext_HAD @ psi
    print(psi)
    psi = Cnot() @ psi
    print(psi)
