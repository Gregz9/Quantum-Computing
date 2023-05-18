from __future__ import annotations
from src.solvers import *
from src.ops import *
from tqdm import tqdm

# The case of spin J=1, solving for eigenvalues using analytical and numerical methods

Vs = np.linspace(0.0, 2.0, 200)
eigvals_an = np.zeros((len(Vs), 3))
eigvals_num_PI = np.zeros((len(Vs), 3))
eigvals_num_QR = np.zeros((len(Vs), 3))

for i, v in enumerate(tqdm(Vs)):
    H = lipkin_H_J1(v)
    eigvals, eigvecs = np.linalg.eig(H)
    eig_num_PI, _ = power_iteration(H, 1000)
    eig_num_QR = QR_solver(H, iters=1000)
    perm = eigvals.argsort()
    perm_num_PI = eig_num_PI.argsort()
    perm_num_QR = eig_num_QR.argsort()
    eigvals_an[i] = eigvals[perm]
    eigvals_num_PI[i] = eig_num_PI[perm_num_PI]
    eigvals_num_QR[i] = eig_num_QR[perm_num_QR]

""" 
In order for us to be able to use the VQE, we'll have to rewrite 
the Hamiltonian matrix of the lipkin model first in terms of the 
Pauli operators, and then use the pauli operators to create gate 
equivalences which lead to the hamiltonian being rotated into the 
Z-basis, and thus allowing us to perform the measuremnt of the ex-
pectation value for the lowest eigenstate. We'll do it in a seperate
function, and import it to this file. The methodology used to rewrite 
the Hamiltonian matrix/our ansatz is presented in "Quantum Computing 
for Programmers" written by Robert Hundt, page 252.
"""
