import numpy as np
from src.solvers import *
from src.ops import *
import matplotlib.pyplot as plt

# np.random.seed(271)
# np.random.seed(1338)
np.random.seed(9)
np.set_printoptions(linewidth=200)
angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(1, 8))


# Check if the init of a 4 qubit state works fine
state4 = ansatz_4qubit(angles)

# gates = prep_circuit_lipkin_J2()

# vals = measure_energy_mul(angles, num_shots=1000, v=1.0)
SWAP, CNOT10, H, S, I = Swap(), Cnot(1, 0), Hadamard(), Sgate(), Identity()
Sdag = S.conj().T


measures = []
unitaries = []
U_ZIII = np.kron(I, np.kron(I, np.kron(I, I)))
unitaries.append(U_ZIII)
U_IZII = np.kron(np.kron(I, I) @ SWAP, np.kron(I, I))
unitaries.append(U_IZII)
U_IIZI = U_IZII @ np.kron(I, np.kron(np.kron(I, I) @ SWAP, I))
unitaries.append(U_IIZI)
U_IIIZ = U_IIZI @ np.kron(I, np.kron(I, np.kron(I, I) @ SWAP))
unitaries.append(U_IIIZ)

U_ZIZI = np.kron(CNOT10, np.kron(I, I)) @ np.kron(I, np.kron(SWAP @ np.kron(I, I), I))

U_XX = CNOT10 @ np.kron(H, H)
U_XI = np.kron(H, I)
U_IX = U_XI @ SWAP
U_XXII = np.kron(U_XX, np.kron(I, I))
unitaries.append(U_XXII)
U_XIXI = U_ZIZI @ np.kron(U_XI, U_XI)
unitaries.append(U_XIXI)
U_XIIX = U_ZIZI @ np.kron(U_XI, U_IX)
unitaries.append(U_XIIX)
U_IXXI = U_ZIZI @ np.kron(U_IX, U_XI)
unitaries.append(U_IXXI)
U_IXIX = U_ZIZI @ np.kron(U_IX, U_IX)
unitaries.append(U_IXIX)
U_IIXX = U_IIZI @ np.kron(I, np.kron(I, U_XX))
unitaries.append(U_IIXX)

U_YY = CNOT10 @ np.kron(H @ Sdag, H @ Sdag)
U_YI = np.kron(H @ Sdag, I)
U_IY = U_YI @ SWAP
U_YYII = np.kron(U_YY, np.kron(I, I))
unitaries.append(U_YYII)
U_YIYI = U_ZIZI @ np.kron(U_YI, U_YI)
unitaries.append(U_YIYI)
U_YIIY = U_ZIZI @ np.kron(U_YI, U_IY)
unitaries.append(U_YIIY)
U_IYYI = U_ZIZI @ np.kron(U_IY, U_YI)
unitaries.append(U_IYYI)
U_IYIY = U_ZIZI @ np.kron(U_IY, U_IY)
unitaries.append(U_IYIY)
U_IIYY = U_IIZI @ np.kron(np.kron(I, I), U_YY)
unitaries.append(U_IIYY)


# Try more epochs, or slighlty increase the value of eta. Other parameters do not need changing
# angles, epoch, energy, delta = VQE(0.1, 400, 1000, angles, 1.0, J=2)
# angles, epoch, energy, delta = VQE(5 * 1e-3, 2000, 1000, angles, 1.0, J=2)
# angles, epoch, energy, delta = VQE(0.009, 2000, 1000, angles, 0.0, J=2)
def prepare_state_test(angles, target=None):
    # # angles has the form (theta0, phi0, theta1, phi1, ...)
    I, X, Y, CNOT10, CNOT01 = qubit.I, qubit.X, qubit.Y, qubit.CNOT10, qubit.CNOT01
    state4 = np.zeros(16)
    state4[0] = 1
    # split the angles list into batches of 8
    angles_batches = [angles[i : i + 8] for i in range(0, len(angles), 8)]
    rotations = []
    for angles in angles_batches:
        for i in range(0, len(angles) - 1, 2):
            theta, phi = angles[i], angles[i + 1]
            Rx = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * X
            Ry = np.cos(phi / 2) * I - 1j * np.sin(phi / 2) * Y
            rotations.append(Ry @ Rx)
    print(len(rotations))
    for _ in range(0, len(rotations), 4):
        rotate = np.kron(
            rotations[i],
            np.kron(rotations[i + 1], np.kron(rotations[i + 2], rotations[i + 3])),
        )
        state4 = rotate @ state4
        state4 = np.kron(I, np.kron(I, CNOT10)) @ state4
        state4 = np.kron(I, np.kron(CNOT10, I)) @ state4
        state4 = np.kron(CNOT10, np.kron(I, I)) @ state4

    if target is not None:
        state4 = target
    return state4


def prepare_state(angles, target=None):
    # # angles has the form (theta0, phi0, theta1, phi1, ...)
    I, X, Y, CNOT10, CNOT01 = Identity(), PauliX(), PauliY(), Cnot(1, 0), Cnot(0, 1)
    init_state = np.zeros(16)
    init_state[0] = 1
    rotations = []
    for i in range(0, len(angles) - 1, 2):
        theta, phi = angles[i], angles[i + 1]
        Rx = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * X
        Ry = np.cos(phi / 2) * I - 1j * np.sin(phi / 2) * Y
        rotations.append(Ry @ Rx)

    rotate = np.kron(
        rotations[0], np.kron(rotations[1], np.kron(rotations[2], rotations[3]))
    )
    state1 = rotate @ init_state
    state1 = np.kron(I, np.kron(I, CNOT10)) @ state1
    state1 = np.kron(I, np.kron(CNOT10, I)) @ state1
    state1 = np.kron(CNOT10, np.kron(I, I)) @ state1
    # entangle = np.kron(CNOT10, CNOT01)
    # state = entangle @ state
    # entangle = np.kron(I, toffoli)
    # state = entangle @ state
    # # rotate qubits

    if target is not None:
        state1 = target
    return state1


def get_energy_test(angles, v, number_shots, unitaries, target=None):
    init_state = prepare_state(angles)
    measures = np.zeros((len(unitaries), number_shots))
    for index, U in enumerate(unitaries):
        state4 = init_state
        state4 = U @ state4
        _, measurement, _, _ = measure(state4, number_shots)
        measures[index] = measurement

    exp_vals = np.zeros(len(measures))
    consts_z = 0.5 * np.ones(4)
    consts_x = -0.5 * v * np.ones(6)
    consts_y = 0.5 * v * np.ones(6)
    constants = np.concatenate((consts_z, consts_x, consts_y))
    for index in range(len(exp_vals)):
        counts = [len(np.where(measures[index] == i)[0]) for i in range(16)]
        for outcome, count in enumerate(counts):
            if outcome <= 7:
                exp_vals[
                    index
                ] += count  # the first 8 outcomes correspond to 0 in the first qubit
            elif outcome > 7:
                exp_vals[
                    index
                ] -= count  # the last 8 outcomes correspond to 1 in the first qubit
    exp_val = np.sum(constants * exp_vals) / number_shots
    return exp_val


def get_gradient(angles, v, number_shots, unitaries):
    grad = np.zeros(len(angles))
    for index, angle in enumerate(angles):
        tmp = angles.copy()
        tmp[index] += np.pi / 2
        energy_plus = get_energy_test(tmp, v, number_shots, unitaries)
        tmp[index] -= np.pi
        energy_minus = get_energy_test(tmp, v, number_shots, unitaries)
        grad[index] = (energy_plus - energy_minus) / 2
    return grad


from scipy.optimize import minimize

number_shots = 1_000
v_vals = np.linspace(1.0, 1.1, 1)  # only 2.0
# target = eigvecs[:, 0] # ground state
min_energy_scipy = np.zeros(len(v_vals))
for index, v in enumerate(tqdm(v_vals)):
    angles_start = np.random.uniform(low=0, high=2 * np.pi, size=5 * 8)
    res = minimize(
        fun=get_energy_test,
        jac=get_gradient,
        x0=angles_start,
        args=(v, number_shots, unitaries),
        method="BFGS",
        options={"maxiter": 10000},
        tol=1e-8,
    )
    min_energy_scipy[index] = res.fun
    print(res)

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
# for i in range(len(eigvals_ana[0])):
    # axs.plot(v_vals_ana, eigvals_ana[:, i], label=f"$E_{i}$")
# axs.plot(v_vals_ana, eigvals_ana[:, 0], label=f"$E_{i}$")
axs.set_xlabel(r"$V/\epsilon$")
axs.scatter(v_vals, min_energy_scipy, label="VQE Energy Scipy", marker="o")
axs.set_ylabel(r"$E/\epsilon$")
# axs.legend(loc = 'upper left')
plt.show()

def get_energy(angles, v, number_shots, target = None):
    init_state = prepare_state(angles, target)
    # SWAP, CNOT10, H, S, I = qubit.SWAP, qubit.CNOT10, qubit.H, qubit.S, qubit.I
    SWAP, CNOT10, H, S, I = Swap(), Cnot(1, 0), Hadamard(), Sgate(), Identity()
    Sdag = S.conj().T

    measures = []
    unitaries = []
    U_ZIII = np.kron(I, np.kron(I, np.kron(I, I)))
    unitaries.append(U_ZIII)
    U_IZII = np.kron(np.kron(I, I)@SWAP, np.kron(I, I))
    unitaries.append(U_IZII)
    U_IIZI = U_IZII@np.kron(I, np.kron(np.kron(I, I)@SWAP, I))
    unitaries.append(U_IIZI)
    U_IIIZ = U_IIZI@np.kron(I, np.kron(I, np.kron(I, I)@SWAP))
    unitaries.append(U_IIIZ)

    U_ZIZI = np.kron(CNOT10, np.kron(I, I))@np.kron(I, np.kron(SWAP@np.kron(I, I), I))
    
    U_XX = CNOT10@np.kron(H, H)
    U_XI = np.kron(H, I)
    U_IX = U_XI@SWAP
    U_XXII = np.kron(U_XX, np.kron(I, I))
    unitaries.append(U_XXII)
    U_XIXI = U_ZIZI@np.kron(U_XI, U_XI)
    unitaries.append(U_XIXI)
    U_XIIX = U_ZIZI@np.kron(U_XI, U_IX)
    unitaries.append(U_XIIX)
    U_IXXI = U_ZIZI@np.kron(U_IX, U_XI)
    unitaries.append(U_IXXI)
    U_IXIX = U_ZIZI@np.kron(U_IX, U_IX)
    unitaries.append(U_IXIX)
    U_IIXX = U_IIZI@np.kron(I, np.kron(I, U_XX))
    unitaries.append(U_IIXX)

    U_YY = CNOT10@np.kron(H@Sdag, H@Sdag)
    U_YI = np.kron(H@Sdag, I)
    U_IY = U_YI@SWAP
    U_YYII = np.kron(U_YY, np.kron(I, I))
    unitaries.append(U_YYII)
    U_YIYI = U_ZIZI@np.kron(U_YI, U_YI)
    unitaries.append(U_YIYI)
    U_YIIY = U_ZIZI@np.kron(U_YI, U_IY)
    unitaries.append(U_YIIY)
    U_IYYI = U_ZIZI@np.kron(U_IY, U_YI)
    unitaries.append(U_IYYI)
    U_IYIY = U_ZIZI@np.kron(U_IY, U_IY)
    unitaries.append(U_IYIY)
    U_IIYY = U_IIZI@np.kron(np.kron(I, I), U_YY)
    unitaries.append(U_IIYY)

    for U in unitaries:
        # qubit.set_state(init_state)
        qu_state = U@init_state
        measurement = measure(qu_state,number_shots)
        measures.append(measurement)
    
    exp_vals = np.zeros(len(measures)) 
    consts_z = 0.5*np.ones(4)
    consts_x = -0.5*v*np.ones(6)
    consts_y = 0.5*v*np.ones(6)
    constants = np.concatenate((consts_z, consts_x, consts_y))
    for index in range(len(exp_vals)):
        counts = [len(np.where(measures[index] == i)[0]) for i in range(16)] 
        for outcome, count in enumerate(counts):
            if outcome <= 7:
                exp_vals[index] += count #the first 8 outcomes correspond to 0 in the first qubit
            elif outcome > 7:
                exp_vals[index] -= count #the last 8 outcomes correspond to 1 in the first qubit
    exp_val = np.sum(constants * exp_vals) / number_shots
    return exp_val


def get_gradient(angles, v, number_shots):
    grad = np.zeros(len(angles))
    for index, angle in enumerate(angles):
        tmp = angles.copy()
        tmp[index] += np.pi/2
        energy_plus = get_energy_test(tmp, v, number_shots, unitaries)
        tmp[index] -= np.pi
        energy_minus = get_energy_test(tmp, v, number_shots, unitaries)
        grad[index] = (energy_plus - energy_minus) / 2
    return grad

    return exp_val


# def minimize_energy(v, number_shots, angles_0, learning_rate, max_epochs):
#     angles = angles_0
#     epoch = 0
#     delta_energy = 1
#     energy = get_energy(angles, v, number_shots)
#     energy_iter = []
#     grad = get_gradient(angles_0, v, number_shots)
#     while (epoch < max_epochs) and (np.linalg.norm(grad) > 1e-5):
#         grad = get_gradient(angles, v, number_shots)
#         angles -= learning_rate * grad
#         new_energy = get_energy(angles, v, number_shots)
#         energy_iter.append(new_energy)
#         delta_energy = np.abs(new_energy - energy)
#         energy = new_energy
#         epoch += 1
#     return angles, epoch, (epoch < max_epochs), energy, delta_energy, grad, energy_iter
#
#
# number_shots = 1_000
# learning_rate = 0.01
# max_epochs = 2000
# angles_0 = np.random.uniform(0, np.pi, 8)
# v = 1.0
# angles, epoch, success, energy, delta_energy, grad, energy_iter = minimize_energy(
#     v, number_shots, angles_0, learning_rate, max_epochs
# )
# print( energy)
# plt.plot(energy_iter)
