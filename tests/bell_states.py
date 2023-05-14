from __future__ import annotations
import numpy as np
import math
from src.solvers import *
from src.ops import *
from src.bell_states import *

if __name__ == "__main__":
    # Preparing bell states
    b00 = bell_state(0, 0)
    print("First bell pair: ")
    print("-------------------------------------------------------------")
    print(b00)
    print("-------------------------------------------------------------\n")

    b01 = bell_state(0, 1)
    print("Second bell pair: ")
    print("-------------------------------------------------------------")
    print(b01)
    print("-------------------------------------------------------------\n")

    b10 = bell_state(1, 0)
    print("Third bell pair: ")
    print("-------------------------------------------------------------")
    print(b10)
    print("-------------------------------------------------------------\n")

    b11 = bell_state(1, 1)
    print("Fourth bell pair: ")
    print("-------------------------------------------------------------")
    print(b11)
    print("-------------------------------------------------------------\n")

    print("Performing measurements on the first bell pair")
    print("-------------------------------------------------------------\n")
    psi, outcome, counts, meas_probs = measure(b00, 1000)
    print("State of the system after measurement: ")
    print(psi, "\n")
    print("Observed states during measurements: ")
    print(f"|00> : {counts[0]}")
    print(f"|10> : {0}")
    print(f"|01> : {0}")
    print(f"|11> : {counts[1]} \n")
    print("Observed probabilities for each of the basis states of the bell pair: ")
    print(f"|00> : {meas_probs[0]*100}%")
    print(f"|10> : {0}")
    print(f"|01> : {0}")
    print(f"|11> : {meas_probs[1]*100}%")

    print(
        "\nThose we're measurements performed on the first of the bell pairs, with 1000 shots \n"
        + "As we can observe the state of the qubit after maesurement always collapses to one of the two \n"
        + "possible states out of four states in total. If I explicitly perform measurements on the seperate\n"
        + "qubits of the entangled bell pair, it becomes apparent that when the qubit I choose to measure \n"
        + "collapses into a one of the basis states |0> or |1>, the second qubit has to follow. However, \n"
        + "which state the second qubit collapses to, is already predefined by the collapse of the \n"
        + "first qubit due to the phenomena of entanglement which gives the bell states their special \n"
        + "attributes.\n"
    )

    print("Performing single projective measurements on the first bell pair b00: ")
    print("----------------------------------------------------------------------")
    print(
        "Here we force one of the quibts in the bell pair to collapse to state |0> or state |1> \n"
    )

    print("Measuring first qubit of the entangled pair: ")
    print("-------------------------------------------------------")
    prob, state = projective_measurement(b00, 0, 0)
    print(f"Probability of the wave function collapsing to |0> : {prob}")
    print(f"State of the bell pair after measurement: {state}\n")

    prob, state = projective_measurement(b00, 0, 1)
    print(f"Probability of the wave function collapsing to |1> : {prob}")
    print(f"State of the bell pair after measurement: {state}\n")

    print("Measuring the second qubit of the pair: ")
    print("-------------------------------------------------------")
    prob, state = projective_measurement(b00, 1, 0)
    print(f"Probability of the wave function collapsing to |0> : {prob}")
    print(f"State of the bell pair after measurement: {state}\n")

    prob, state = projective_measurement(b00, 1, 1)
    print(f"Probability of the wave function collapsing to |1> : {prob}")
    print(f"State of the bell pair after measurement: {state}\n")

    print(
        "As we can see, no matter which of the qubits of the bell pair we measure,\n"
        + "the bell pair will always collapse to one the two entangled states, as indicated\n"
        + "by the measurements performed above.\n"
    )
    print(
        "---------------------------------------------------------------------------------"
    )

    print(
        "What happens then if we apply the CNOT gate and the Hadamard gate to the first bell pair b00?\n\n"
    )

    print("Constructing and applying the Controlled-NOT gate to the first bell pair: ")
    print(
        "---------------------------------------------------------------------------------"
    )

    CNOT = Cnot()
    print(CNOT)

    print("Result of applying the CNOT gate to b00: ")
    CNOT_b00 = CNOT @ b00
    print(CNOT_b00)

    print(
        "Result of applying the Hadamard gate to b00, after having applied the CNOT: "
    )
    HCNOT_b00 = np.kron(Hadamard(), Identity()) @ CNOT_b00
    HCNOT_b00 = np.where(HCNOT_b00 > 1e-10, HCNOT_b00, 0)
    print(HCNOT_b00)

    print("Result of applying the Hadamard gate to b00: ")
    H_b00 = np.kron(Hadamard(), np.eye(2)) @ b00
    print(H_b00)

    print(
        "Result of applying the CNOT gate to b00, after having applied the Hadamard gate: "
    )
    CNOTH_b00 = CNOT @ H_b00
    print(CNOTH_b00)
