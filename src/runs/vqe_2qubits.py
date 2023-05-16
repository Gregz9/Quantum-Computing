from __future__ import annotations
import numpy as np
import math
from src.solvers import *
from src.ops import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm

basis0 = np.array([1, 0])
basis1 = np.array([0, 1])

op0 = np.kron(basis0, Identity())
op1 = np.kron(basis1, Identity())

H = hamiltonian_2qubit(1.0, 2.0, 3.0, np.array([0.0, 2.5, 6.5, 7.0]))
