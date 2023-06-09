import numpy as np
from src.solvers import *
from src.ops import *
import matplotlib.pyplot as plt

# np.random.seed(271)
# np.random.seed(1338)
np.random.seed(9)
np.set_printoptions(linewidth=200)
angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(1, 8))


eta = 0.002
mnt = 0.02
epochs = 1000
num_shots = 1000

_, _, energy, _ = VQE(eta, epochs, num_shots, angles, 1.0, J=2)

print(energy)
