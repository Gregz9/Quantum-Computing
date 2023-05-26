import numpy as np


def create_hamiltonian_matrix(N, W):
    matrix_size = 2**N
    hamiltonian = np.zeros((matrix_size, matrix_size))

    for i in range(matrix_size):
        for j in range(matrix_size):
            binary_i = format(i, f"0{N}b")
            binary_j = format(j, f"0{N}b")
            term = -N

            for p in range(N):
                for q in range(N):
                    if p != q:
                        spin_operator = 1

                        if binary_i[p] == "0" and binary_i[q] == "1":
                            spin_operator *= 1  # Jp+Jq-
                        elif binary_i[p] == "1" and binary_i[q] == "0":
                            spin_operator *= 1  # Jq+Jp-
                        else:
                            spin_operator = 0

                        term += spin_operator

            hamiltonian[i, j] = (1 / 2) * W * term

    return hamiltonian


N = 4  # Number of particles
W = 1.0  # Coefficient W

hamiltonian_matrix = create_hamiltonian_matrix(N, W)
print(hamiltonian_matrix)
