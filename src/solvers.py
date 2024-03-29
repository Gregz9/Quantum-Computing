from __future__ import annotations
import numpy as np
import math
import cmath
from typing import Iterable, List, Tuple
from scipy.optimize import minimize
from src.ops import *
from tqdm import tqdm


"""
This file contains all the algorithms used for approximating/solving the 
eigenvalue problems associated with each of the scripts found in the "runs"
directory, starting with the extremely simple one-body Hamiltonian matrix, 
and ending at the Lipkin model with spin J=2. It also contains the methods
for the computation of the von neumann entropy for the two-body hamiltonian 
(not Lipkin model with J=1). The code here contains little comments, however 
the names should indicate which problem the methods correspond to.
"""

_PAULI_X = PauliX()
_PAULI_Y = PauliY()
_PAULI_Z = PauliZ()
_HADAMARD = Hadamard()
_SGATE = Sgate()


def hamilitonian(dim=2, e0=0, e1=1.0, Xdiag=1.0, Xnondiag=0.0, lam=1.0):
    Hamilitonian = np.zeros((dim, dim))
    Hamilitonian[0, 0] = e0 + (lam * Xdiag)
    Hamilitonian[0, 1] = lam * Xnondiag
    Hamilitonian[1, 0] = lam * Xnondiag
    Hamilitonian[1, 1] = e1 - (lam * Xdiag)

    return Hamilitonian


def hamiltonian_1qubit(
    dim=2, e0=0.0, e1=0.0, V11=1.0, V12=1.0, V21=1.0, V22=1.0, lam=1.0
):
    eps = (e0 + e1) / 2
    omega = (e0 - e1) / 2
    c = (V11 + V22) / 2
    omega_z = (V11 - V22) / 2
    omega_x = V21

    H0 = eps * Identity() + omega * _PAULI_Z
    H1 = c * Identity() + omega_z * _PAULI_Z + omega_x * _PAULI_X
    H = H0 + lam * H1
    return H, [eps, c, omega, omega_z, omega_x]


def hamiltonian_2qubit(lmbd, Hx, Hz, H0_list):
    H0 = np.diag(H0_list)
    HI = Hx * np.kron(PauliX(), PauliX()) + Hz * np.kron(PauliZ(), PauliZ())
    return H0 + lmbd * HI


def lipkin_H_J1(v):
    H = np.array([[1.0, 0.0, -v], [0.0, 0.0, 0.0], [-v, 0.0, -1.0]])
    return H


def lipkin_H_J1_Pauli(v, w=0, full=False):
    I = Identity()
    X = PauliX()
    Y = PauliY()
    Z = PauliZ()
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)
    ZI = np.kron(Z, I)
    IZ = np.kron(I, Z)
    N = 2
    # The lipkin hamiltonian for spin J=1, particles N=2
    if not full:
        H = (1 / 2) * (ZI + IZ) - (v / 2) * (XX - YY)
    else:
        H2 = -(w / 2) * (-np.diag([2] * 4) + XX + YY)
        H = (1 / 2) * (ZI + IZ) - (v / 2) * (XX - YY) + H2

    return H


def lipkin_H_J2_Pauli(v, w=0, full=False):
    I = Identity()
    X = PauliX()
    Y = PauliY()
    Z = PauliZ()
    ZIII = np.kron(Z, np.kron(I, np.kron(I, I)))
    IZII = np.kron(I, np.kron(Z, np.kron(I, I)))
    IIZI = np.kron(I, np.kron(I, np.kron(Z, I)))
    IIIZ = np.kron(I, np.kron(I, np.kron(I, Z)))

    XXII = np.kron(X, np.kron(X, np.kron(I, I)))
    XIXI = np.kron(X, np.kron(I, np.kron(X, I)))
    XIIX = np.kron(X, np.kron(I, np.kron(I, X)))
    IXXI = np.kron(I, np.kron(X, np.kron(X, I)))
    IXIX = np.kron(I, np.kron(X, np.kron(I, X)))
    IIXX = np.kron(I, np.kron(I, np.kron(X, X)))

    YYII = np.kron(Y, np.kron(Y, np.kron(I, I)))
    YIYI = np.kron(Y, np.kron(I, np.kron(Y, I)))
    YIIY = np.kron(Y, np.kron(I, np.kron(I, Y)))
    IYYI = np.kron(I, np.kron(Y, np.kron(Y, I)))
    IYIY = np.kron(I, np.kron(Y, np.kron(I, Y)))
    IIYY = np.kron(I, np.kron(I, np.kron(Y, Y)))

    N1 = np.kron((1 / 2) * (I - Z), np.kron(I, np.kron(I, I)))
    N2 = np.kron(I, np.kron((1 / 2) * (I - Z), np.kron(I, I)))
    N3 = np.kron(I, np.kron(I, np.kron((1 / 2) * (I - Z), I)))
    N4 = np.kron(I, np.kron(I, np.kron(I, (1 / 2) * (I - Z))))
    N = N1 + N2 + N3 + N4

    # Jm = (X - 1j * Y) / np.sqrt(2)
    # Jp = (X + 1j * Y) / np.sqrt(2)

    # JpJmII = np.kron(Jp, np.kron(Jm, np.kron(I, I)))
    # JmJpII = np.kron(Jm, np.kron(Jp, np.kron(I, I)))
    # JpIJmI = np.kron(Jp, np.kron(I, np.kron(Jm, I)))
    # JmIJpI = np.kron(Jm, np.kron(I, np.kron(Jp, I)))
    # JpIIJm = np.kron(Jp, np.kron(I, np.kron(I, Jm)))
    # JmIIJp = np.kron(Jm, np.kron(I, np.kron(I, Jp)))
    # IJpJmI = np.kron(I, np.kron(Jp, np.kron(Jm, I)))
    # IJmJpI = np.kron(I, np.kron(Jm, np.kron(Jp, I)))
    # IJpIJm = np.kron(I, np.kron(Jp, np.kron(I, Jm)))
    # IJmIJp = np.kron(I, np.kron(Jm, np.kron(I, Jp)))
    # IIJpJm = np.kron(I, np.kron(I, np.kron(Jp, Jm)))
    # IIJmJp = np.kron(I, np.kron(I, np.kron(Jm, Jp)))

    if not full:
        H = (
            (1 / 2) * (ZIII + IZII + IIZI + IIIZ)
            - (v / 2) * (XXII + XIXI + XIIX + IXXI + IXIX + IIXX)
            + (v / 2) * (YYII + YIYI + YIIY + IYYI + IYIY + IIYY)
        )
    elif full:
        H = (
            (1 / 2) * (ZIII + IZII + IIZI + IIIZ)
            - (v / 2) * (XXII + XIXI + XIIX + IXXI + IXIX + IIXX)
            + (v / 2) * (YYII + YIYI + YIIY + IYYI + IYIY + IIYY)
            + (w / 2)
            * (
                -N
                + XXII
                + XIXI
                + XIIX
                + IXXI
                + IXIX
                + IIXX
                + YYII
                + YIYI
                + YIIY
                + IYYI
                + IYIY
                + IIYY
            )
        )

    return H


def trace_first(state):
    density = np.outer(state, state.conj())
    basis0 = np.array([1, 0])
    basis1 = np.array([0, 1])
    qubit0_0 = np.kron(basis0, Identity())
    qubit0_1 = np.kron(basis1, Identity())
    return (
        qubit0_0.conj() @ density @ qubit0_0.T + qubit0_1.conj() @ density @ qubit0_1.T
    )


def trace_second(state):
    density = np.outer(state, state.conj())
    basis0 = np.array([1, 0])
    basis1 = np.array([0, 1])
    qubit1_0 = np.kron(Identity(), basis0)
    qubit1_1 = np.kron(Identity(), basis1)
    return (
        qubit1_0.conj() @ density @ qubit1_0.T + qubit1_1.conj() @ density @ qubit1_1.T
    )


def analytical_solver(mat):
    EigValues, EigVectors = np.linalg.eig(mat)
    permute = EigValues.argsort()
    Eigvalues, EigVectors = EigValues[permute], EigVectors[permute]

    return Eigvalues, EigVectors


def normalize(x):
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n


def power_iteration(a, num_iterations, seed=11337) -> (np.ndarray, np.ndarray):
    np.random.seed(seed)
    n = len(a)
    eigenvalues = []
    eigenvectors = []

    for i in range(n):
        x = np.random.rand(n)

        for j in range(num_iterations):
            x = np.dot(a, x)
            eigenvalue, x = normalize(x)

        eigenvectors.append(x)
        eigenvalues.append(eigenvalue)
        a = a - eigenvalue * np.outer(x, x)

    return np.array(eigenvalues), np.array(eigenvectors)


def QR_solver(H, delta=1e-6, iters=100):
    m, n = H.shape
    Q = np.eye(m)

    for iteration in range(iters):
        Q, R = np.linalg.qr(H)
        H = Q.T @ H @ Q

        # Check convergence
        off_diag_sum = np.sum(np.abs(H - np.diag(np.diag(H))))
        if off_diag_sum < delta:
            break

    eigenvalues = np.diag(H)
    return eigenvalues


def von_neumann(eigvals, eigvecs, ind):
    # The eigenvalues and eigenvectors have to be sorted from lowest to highest
    trace_op = trace_first if ind == 0 else trace_second
    entropy = np.zeros(eigvecs.shape[1])
    for i in range(eigvecs.shape[1]):
        sub_den = trace_op(eigvecs[:, i])
        sub_den = np.linalg.eigvalsh(sub_den)
        sub_den = np.ma.masked_equal(sub_den, 0).compressed()
        entropy[i] = -np.sum(sub_den * np.log2(sub_den))
    return entropy


def ansatz_1qubit(theta, phi) -> np.ndarray:
    Rx = np.cos(theta * 0.5) * Identity() - 1j * np.sin(theta * 0.5) * _PAULI_X
    Ry = np.cos(phi * 0.5) * Identity() - 1j * np.sin(phi * 0.5) * _PAULI_Y
    basis0 = np.array([1, 0])
    return Ry @ Rx @ basis0, Rx, Ry, basis0


def ansatz_2qubit(
    angles: np.ndarray = np.array([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])
) -> np.ndarray:
    basis0 = np.array([1, 0])
    basis00 = np.kron(basis0, basis0)

    Rx = np.cos(angles[0] * 0.5) * Identity() - 1j * np.sin(angles[0] * 0.5) * _PAULI_X
    Ry = np.cos(angles[1] * 0.5) * Identity() - 1j * np.sin(angles[1] * 0.5) * _PAULI_Y

    RyRx_q0 = Ry @ Rx
    RyRx_q0 = np.kron(RyRx_q0, Identity())
    init_state = RyRx_q0 @ basis00

    Rx = np.cos(angles[2] * 0.5) * Identity() - 1j * np.sin(angles[2] * 0.5) * _PAULI_X
    Ry = np.cos(angles[3] * 0.5) * Identity() - 1j * np.sin(angles[3] * 0.5) * _PAULI_Y

    RyRx_q1 = Ry @ Rx
    RyRx_q1 = np.kron(Identity(), RyRx_q1)
    init_state = RyRx_q1 @ init_state
    init_state = Cnot(0, 1) @ init_state

    return init_state


def ansatz_4qubit(angles: np.ndarray):
    """
    We assume that the array of angles is organized into batches, which here
    have the shape an input array of a neural network would have. Meaning
    m rows indicating m data samples, with n features, here indicating n angles.
    """
    state4 = np.kron(
        state(alpha=1.0),
        np.kron(state(alpha=1.0), np.kron(state(alpha=1.0), state(alpha=1.0))),
    )
    angles = np.array([angles[i : i + 8] for i in range(0, len(angles), 8)])
    rots = []
    for i in range(angles.shape[0]):
        for j in range(0, angles.shape[1] - 1, 2):
            theta, phi = angles[i, j], angles[i, j + 1]
            Rx = np.cos(theta * 0.5) * Identity() - 1j * np.sin(theta * 0.5) * PauliX()
            Ry = np.cos(phi * 0.5) * Identity() - 1j * np.sin(phi * 0.5) * PauliY()
            rots.append(Ry @ Rx)
    rots2 = np.stack(rots)
    for r in range(0, len(rots), 4):
        rot = np.kron(
            rots2[r], np.kron(rots2[r + 1], np.kron(rots2[r + 2], rots2[r + 3]))
        )
        state4 = rot @ state4
        state4 = np.kron(np.eye(2), np.kron(np.eye(2), Cnot(1, 0))) @ state4
        state4 = np.kron(np.eye(2), np.kron(Cnot(1, 0), np.eye(2))) @ state4
        state4 = np.kron(Cnot(1, 0), np.eye(4)) @ state4

    return state4


def measure_energy_J1(
    angles=np.array([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]), v=1.0, shots=1
):
    init_ansatz = ansatz_2qubit(angles)
    H = Hadamard()

    post_state_ZI, measure_ZI, counts_ZI, obs_probs_ZI = measure(init_ansatz, shots)

    measure_IZ = Swap() @ init_ansatz
    post_state_IZ, measure_IZ, counts_IZ, obs_probs_IZ = measure(measure_IZ, shots)

    measure_XX = Cnot(1, 0) @ np.kron(H, H) @ init_ansatz
    post_state_XX, measure_XX, counts_XX, obs_probs_XX = measure(measure_XX, shots)

    measure_YY = (
        Cnot(1, 0) @ np.kron(H @ Sgate().conj().T, H @ Sgate().conj().T) @ init_ansatz
    )
    post_state_YY, measure_YY, counts_YY, obs_probs_YY = measure(measure_YY, shots)

    exp_vals = np.zeros(4)
    measures = np.array([measure_IZ, measure_ZI, measure_XX, measure_YY])
    consts = np.array([1 / 2, 1 / 2, -v / 2, v / 2])
    for i in range(exp_vals.shape[0]):
        counts = [len(np.where(measures[i] == j)[0]) for j in range(4)]
        exp_vals[i] = counts[0] + counts[1] - counts[2] - counts[3]

    return np.sum(exp_vals * consts) / shots


def prep_circuit_lipkin_J2():
    I = np.eye(2)
    ZIII = np.eye(16)
    IZII = np.kron(np.eye(4) @ Swap(), np.eye(4))
    # In order to rotate the basis to ZIII from IIZI we have to apply the Swap gate twice.
    # This is equivalent to applying the Swap gate twice. This is equivalent to applying
    # this operator to The previous one we've computed
    IIZI = IZII @ np.kron(
        np.eye(2), np.kron(np.kron(np.eye(2), np.eye(2)) @ Swap(), np.eye(2))
    )
    IIIZ = IIZI @ np.kron(
        np.eye(2), np.kron(np.eye(2), np.kron(np.eye(2), np.eye(2)) @ Swap())
    )

    ZIZI = np.kron(Cnot(1, 0), np.eye(4)) @ np.kron(
        I, np.kron(Swap() @ np.kron(I, I), I)
    )
    # Now we have to apply a set of gates to the other operators we have used to rewrite our hami-
    # ltonian matrix
    XXII = np.kron(Cnot(1, 0) @ np.kron(Hadamard(), Hadamard()), np.eye(4))
    # In the upcoming state, we're going to first rotate the basis into the state ZIZI
    # And then rotate it into ZIII. However as you have seen, we have done the latter,
    # and need only to mulitply the former operator with the latter we have created.
    XIXI = ZIZI @ np.kron(
        np.kron(Hadamard(), np.eye(2)), np.kron(Hadamard(), np.eye(2))
    )
    XIIX = ZIZI @ np.kron(
        np.kron(Hadamard(), np.eye(2)),
        np.kron(Hadamard(), np.eye(2)) @ Swap(),
    )
    IXXI = ZIZI @ (
        np.kron(np.kron(Hadamard(), np.eye(2)) @ Swap(), np.kron(Hadamard(), np.eye(2)))
    )
    IXIX = ZIZI @ (
        np.kron(
            np.kron(Hadamard(), np.eye(2)) @ Swap(),
            np.kron(Hadamard(), np.eye(2)) @ Swap(),
        )
    )
    IIXX = IIZI @ np.kron(I, np.kron(I, Cnot(1, 0) @ np.kron(Hadamard(), Hadamard())))

    # Rotating the Y-basis
    YYII = np.kron(
        Cnot(1, 0)
        @ np.kron(Hadamard() @ Sgate().conj().T, Hadamard() @ Sgate().conj().T),
        np.eye(4),
    )

    YIYI = ZIZI @ np.kron(
        np.kron(Hadamard() @ Sgate().conj().T, np.eye(2)),
        np.kron(Hadamard() @ Sgate().conj().T, np.eye(2)),
    )
    YIIY = ZIZI @ np.kron(
        np.kron(Hadamard() @ Sgate().conj().T, np.eye(2)),
        np.kron(Hadamard() @ Sgate().conj().T, np.eye(2)) @ Swap(),
    )
    IYYI = ZIZI @ np.kron(
        np.kron(Hadamard() @ Sgate().conj().T, np.eye(2)) @ Swap(),
        np.kron(Hadamard() @ Sgate().conj().T, np.eye(2)),
    )
    IYIY = ZIZI @ np.kron(
        np.kron(Hadamard() @ Sgate().conj().T, np.eye(2)) @ Swap(),
        np.kron(Hadamard() @ Sgate().conj().T, np.eye(2)) @ Swap(),
    )
    IIYY = IIZI @ np.kron(
        np.eye(4),
        Cnot(1, 0)
        @ np.kron(Hadamard() @ Sgate().conj().T, Hadamard() @ Sgate().conj().T),
    )
    return [
        ZIII,
        IZII,
        IIZI,
        IIIZ,
        XXII,
        XIXI,
        XIIX,
        IXXI,
        IXIX,
        IIXX,
        YYII,
        YIYI,
        YIIY,
        IYYI,
        IYIY,
        IIYY,
    ]


def measure_energy_1q(angles=np.array([np.pi / 2, np.pi / 2]), lmb=1.0, shots=1):
    _, elements = hamiltonian_1qubit(
        2, e0=0.0, e1=4.0, V11=3, V12=0.2, V21=0.2, V22=-3, lam=lmb
    )
    # elements have the following structure: [eps, c, omega, omega_z, omega_x]

    # In order to measure the estimated energy, we'll have to rewrite all
    # elements of the Hamiltonian into the pauli Z basis, as it's the only
    # of the pauli matrices with vectors |0> and |1> forming its basis.

    init_state, _, _, _ = ansatz_1qubit(angles[0], angles[1])
    _, measure_z, counts, obs_probs_z = measure(init_state, shots)

    init_state, _, _, _ = ansatz_1qubit(angles[0], angles[1])
    measure_x = Hadamard() @ init_state
    _, measure_x, counts, obs_probs_x = measure(measure_x, shots)

    exp_val_z = (
        (elements[2] + lmb * elements[3]) * (shots - 2 * np.sum(measure_z)) / shots
    )
    exp_val_x = lmb * elements[4] * (shots - 2 * np.sum(measure_x)) / shots
    exp_val_i = elements[0] + elements[1] * lmb
    exp_val = exp_val_z + exp_val_x + exp_val_i

    return exp_val


def measure_energy_2q(
    angles: np.ndarray = np.array([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2]),
    lmb: float = 1.0,
    shots: int = 1000,
):
    Hx = 2.0
    Hz = 3.0
    E_non = np.array([0.0, 2.5, 6.5, 7.0])
    signs = (1 / 4) * np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    )
    # The Hamiltonian introduced in part d of the project can be rewriten
    # using first projection operators, which in turn allows us to use
    # the Identity and Pauli Z matrices to rewrite the hamiltonian of this
    # 2 qubit system.
    coeffs = np.sum(signs * E_non, axis=1)
    init_ansatz = ansatz_2qubit(angles)

    post_state_ZI, measure_ZI, counts_ZI, obs_probs_IZ = measure(init_ansatz, shots)

    measure_IZ = Swap() @ init_ansatz
    post_state_IZ, measure_IZ, counts_IZ, obs_probs_IZ = measure(measure_IZ, shots)

    measure_ZZ = Cnot(1, 0) @ init_ansatz
    post_state_ZZ, measure_ZZ, counts_ZZ, obs_probs_ZZ = measure(measure_ZZ, shots)

    measure_XX = Cnot(1, 0) @ np.kron(Hadamard(), Hadamard()) @ init_ansatz
    post_state_XX, measure_XX, counts_XX, obs_probs_XX = measure(measure_XX, shots)

    exp_vals = np.zeros(4)
    measures = np.array([measure_IZ, measure_ZI, measure_ZZ, measure_XX])
    consts = np.array([coeffs[1], coeffs[2], coeffs[3] + lmb * Hz, lmb * Hx])
    for i in range(exp_vals.shape[0]):
        counts = [len(np.where(measures[i] == j)[0]) for j in range(4)]
        exp_vals[i] = counts[0] + counts[1] - counts[2] - counts[3]

    return coeffs[0] + np.sum(consts * exp_vals) / shots


def measure_energy_mul(angles, v, num_shots, w=None, full=False):
    """Method used for measurements of expectation values used for computations associated
    with the lipkin model only
    """
    gates = prep_circuit_lipkin_J2()
    state4 = ansatz_4qubit(angles)
    measurements = np.zeros((len(gates), num_shots))
    for i, op in enumerate(gates):
        state_meas = op @ state4
        post_state, measurement, counts, obs_probs = measure(state_meas, num_shots)
        measurements[i] = measurement

    exps = np.zeros(len(measurements))

    if full:
        consts = np.concatenate(
            (
                0.5 * np.ones(4),
                -0.5 * v * np.ones(6),
                0.5 * v * np.ones(6),
                0.5 * w * np.ones(12),
            )
        )
    else:
        consts = np.concatenate(
            (0.5 * np.ones(4), -0.5 * v * np.ones(6), 0.5 * v * np.ones(6))
        )

    for i in range(len(exps)):
        counts = [len(np.where(measurements[i] == j)[0]) for j in range(16)]
        for out, count in enumerate(counts):
            if out <= 7:
                exps[i] += count
            elif out > 7:
                exps[i] -= count
    if full:
        # The following computation is used for the approximation of the ground state of a Hamiltonian
        # with W > 0. To achieve a good approximation from only one measurement, we need to seed.
        np.random.seed(7)
        N1 = np.kron((1 / 2) * (I - Z), np.kron(I, np.kron(I, I)))
        N2 = np.kron(I, np.kron((1 / 2) * (I - Z), np.kron(I, I)))
        N3 = np.kron(I, np.kron(I, np.kron((1 / 2) * (I - Z), I)))
        N4 = np.kron(I, np.kron(I, np.kron(I, (1 / 2) * (I - Z))))
        N = N1 + N2 + N3 + N4

        N_exp = np.sum(np.abs(N @ state4) ** 2)
        return (
            np.sum(
                np.hstack(
                    (
                        consts[:4] * exps[:4],
                        consts[4:16] * exps[4:] + consts[16:] * exps[4:],
                    )
                )
            )
            / num_shots
            - w * 0.5 * N_exp
        )  # expectation value of the number operator
    else:
        return np.sum(consts * exps) / num_shots


def chose_measurement(length, J=0):
    if J == 2:
        measurement_operation = measure_energy_mul
    elif length == 2 and J == 0:
        measurement_operation = measure_energy_1q
    elif length == 2 and J > 0:
        measurement_operation = measure_energy_J1
    elif length == 4 and J == 0:
        measurement_operation = measure_energy_2q
    elif length == 4 and J > 0:
        measurement_operation = measure_energy_J1

    return measurement_operation


# Generalized version of VQE
def VQE(eta, epochs, num_shots, init_angles, lmbd, J=0):
    """
    One of the two main version of the VQE alogrithm used for approximating the
    ground state energy of the hamiltonians presented in this project.
    """

    measure_energy = chose_measurement(len(init_angles), J)
    angles = init_angles.copy()
    energy = measure_energy(angles, lmbd, num_shots)
    for epoch in range(epochs):
        grad = np.zeros((angles.shape))
        for i in range(angles.shape[0]):
            angles_tmp = angles.copy()
            grad[i] = calc_grad(angles_tmp, i, lmbd, num_shots, J)
        angles -= eta * grad
        new_energy = measure_energy(angles, lmbd, num_shots)
        delta_energy = np.abs(new_energy - energy)
        # if delta_energy < 1e-10:
        # break

        energy = new_energy

    return angles, epoch, energy, delta_energy


def VQE_scipy(
    measure_method,
    v_params,
    angles_dims,
    shots,
    low,
    high,
    method="Powell",
    w_params=None,
    full=False,
):
    """VQE algorithm enchanced with the minimize method from the scipy library.
    compared to the other VQE method which coputes the gradient from scratch, this
    method utilizes "minimize" for this purpose. Compared to the other generalized
    VQE algorithm, it also takes in a measure method as a parameter. For our puroposes
    this method is called "measure_energy_mul", which is used for the measurements
    associated with the lipkin model"""

    min_energy = np.zeros(len(v_params))
    for i, v in enumerate(tqdm(v_params)):
        init_angles = np.random.uniform(low=low, high=high, size=angles_dims)
        res = minimize(
            fun=measure_method,
            jac=get_gradient,
            x0=init_angles,
            args=(v_params[i], shots)
            if not full
            else (v_params[i], shots, w_params[i], full),
            method=method,
            options={"maxiter": 10000},
            tol=1e-11,
        )
        min_energy[i] = res.fun
    return min_energy


def VQE_momentum(eta, mnt, epochs, num_shots, init_angles, lmbd, J=0):
    measure_energy = chose_measurement(len(init_angles), J)
    angles = init_angles
    energy = measure_energy(init_angles, lmbd, num_shots)
    change = np.zeros((angles.shape))
    for epoch in range(epochs):
        grad = np.zeros((angles.shape))
        for i in range(angles.shape[0]):
            angles_temp = angles.copy()
            grad[i] = calc_grad(angles_temp, i, lmbd, num_shots, J)
        new_change = eta * grad + mnt * change
        angles -= new_change
        change = new_change
        new_energy = measure_energy(angles, lmbd, num_shots)
        delta_energy = np.abs(new_energy - energy)
        energy = new_energy
        if delta_energy < 1e-7:
            break

    return angles, epoch, energy, delta_energy


def VQE_Adam(eta, beta1, beta2, epochs, num_shots, init_angles, lmbd, J=0):
    measure_energy = chose_measurement(len(init_angles), J)
    angles = init_angles
    energy = measure_energy(init_angles, lmbd, num_shots)
    t = 0
    for epoch in tqdm(range(epochs)):
        m = 0.0
        v = 0.0
        t += 1
        grad = np.zeros((angles.shape))
        for i in range(angles.shape[0]):
            angles_temp = angles.copy()
            grad[i] = calc_grad(angles_temp, i, lmbd, num_shots, J)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        m_term = m / (1.0 - beta1**t)
        v_term = v / (1.0 - beta2**t)
        angles -= eta * m_term / (np.sqrt(v_term) + 1e-7)

        new_energy = measure_energy(angles, lmbd, num_shots)
        delta_energy = np.abs(new_energy - energy)
        energy = new_energy
        if delta_energy < 1e-7:
            break
    return angles, epoch, energy


def calc_grad(angles, ind, lmbd, num_shots, J=0):
    # Used for computation of the gradient in the "VQE" method presented above
    measure_energy = chose_measurement(angles.shape[0], J)
    angles[ind] += np.pi / 2
    ener_pl = measure_energy(angles, lmbd, num_shots)
    angles[ind] -= np.pi
    ener_min = measure_energy(angles, lmbd, num_shots)
    grad = (ener_pl - ener_min) / 2
    return grad


def get_gradient(angles, v, number_shots, w=None, full=False):
    # Used for minimazing the expectation value in the function "VQE_scipy"
    unitaries = prep_circuit_lipkin_J2()
    grad = np.zeros(len(angles))
    for index, angle in enumerate(angles):
        tmp = angles.copy()
        tmp[index] += np.pi / 2
        energy_plus = measure_energy_mul(tmp, v, number_shots, w, full)
        tmp[index] -= np.pi
        energy_minus = measure_energy_mul(tmp, v, number_shots, w, full)
        grad[index] = (energy_plus - energy_minus) / 2
    return grad


def VQE_naive(H, inter):
    angles = np.arange(0, 180, inter)
    n = np.size(angles)
    ExpectationValues = np.zeros((n, n))
    Energies = np.zeros((n, n))
    EigValues, _ = analytical_solver(H)
    for i in range(n):
        theta = np.pi * angles[i] / 180.0
        for j in range(n):
            phi = np.pi * angles[j] / 180.0
            new_basis, _, _, _ = new_basis_1q(theta, phi)
            Energy = new_basis.conj().T @ H @ new_basis
            Ediff = abs(np.real(EigValues[0] - Energy))
            ExpectationValues[i, j] = Ediff
            Energies[i, j] = Energy

    print(np.min(ExpectationValues))
    ind = np.argmin(ExpectationValues)
    print(Energies.flatten()[ind])


def get_energy(H, theta, phi, collapse_to=0):
    basis = np.zeros(2)
    basis[collapse_to] = 1
    basis = RotationY(phi) @ RotationX(theta) @ basis
    return basis.conj().T @ H @ basis
