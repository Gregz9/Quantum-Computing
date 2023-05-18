from __future__ import annotations
import numpy as np
import math
from src.solvers import *
from src.ops import * 

if __name__ == "__main__":
    # e00, e10, e01, e11 = 0.0, 2.5, 6.5, 7.0
    # Hx, Hz = 2.0, 3.0
    # mat = np.zeros((4, 4))
    # mat[0, 0], mat[1, 1], mat[2, 2], mat[3, 3] = e00 + Hz, e10 - Hz, e01 - Hz, e11 + Hz
    # mat[0, 3], mat[1, 2], mat[2, 1], mat[3, 0] = Hx, Hx, Hx, Hx
    #
    # eig_vals, eig_vecs = analytical_solver(mat)
    # print(eig_vals)
    #
    # VQE_naive_2q(mat, 10)

    hamil = hamilitonian(dim=2, e0=0.0, e1=4.0, Xdiag=3, Xnondiag=0.2)
    # hamil2 = Pauli_hamiltionian(dim=2, e0=0.0, e1=4.0, Xdiag=3.0, Xnondiag=0.2, lam=0.5)
    # eig_vals, eig_vecs = analytical_solver(hamil)
    # print(eig_vals)
    # eig_vals2, eig_vecs2 = analytical_solver(hamil)
    # VQE_naive(hamil, 5)
    # VQE_1q(hamil)

    epsilon = 2
    V = 3
    H0 = epsilon*(np.kron(PauliZ(), Identity()) + np.kron(Identity(), PauliZ()))
    H1 = (V/2)*(np.kron(PauliX(), PauliX()) - np.kron(PauliY(), PauliY()))
    
    Hamiltonian  = H0 + H1
    reduced_Hamiltonian = np.delete(np.delete(Hamiltonian, 1, 1), 1, 0)

    eigvals_H, _ = analytical_solver(Hamiltonian)
    eigvals_RH, _ = analytical_solver(reduced_Hamiltonian)

    print(eigvals_H)
    print(eigvals_RH)


