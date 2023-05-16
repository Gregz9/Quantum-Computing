from __future__ import annotations
import numpy as np
import math
from scipy.optimize import minimize
from tqdm import tqdm
from src.solvers import *
from src.ops import *

basis0 = np.array([1.0, 0.0])
basis00 = np.kron(basis0, basis0)

print(basis00)
