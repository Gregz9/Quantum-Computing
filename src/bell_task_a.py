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
    print(f"|11> : {meas_probs[1]*100}%\n")

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
    print("The CNOT(0, 1)-gate represented as a matrix: ")
    print(CNOT.real, "\n")

    print("The bell state prior to the application of CNOT(0, 1)-gate: ")
    print(b00, "\n")

    print("The bell state after the application CNOT(0, 1)-gate: ")
    CNOT_b00 = CNOT @ b00
    print(CNOT_b00, "\n")

    print("Measurements after application of CNOT(0, 1)-gate")
    print("-------------------------------------------------------------\n")
    psi, outcome, counts, meas_probs = measure(CNOT_b00, 1000)
    print("State of the system after measurement: ")
    print(psi, "\n")
    print("Observed states during measurements: ")
    print(f"|00> : {counts[0]}")
    print(f"|10> : {0}")
    print(f"|01> : {counts[1]}")
    print(f"|11> : {0} \n")
    print("Observed probabilities for each of the basis states of the bell pair: ")
    print(f"|00> : {meas_probs[0]*100}%")
    print(f"|10> : {0}")
    print(f"|01> : {meas_probs[1]*100}%")
    print(f"|11> : {0} \n\n")

    print("Constructing and applying the Hadamard gate to the first bell pair: ")
    print(
        "---------------------------------------------------------------------------------"
    )

    HADAMARD = Hadamard()
    print("The HADAMARD represented as a matrix: ")
    print(HADAMARD.real, "\n")

    print(
        "Before proceeding, we have to extend our Hadamard matrix by a tensor product with a\n"
        "2x2 identity matrix, in order to apply it to the bell state. Here, we will applying the\n"
        "Hadamard to the first qubit of the pair, thus we tensor the Hadamard gate from the right.\n"
    )

    q0_Hadamard = np.kron(HADAMARD, Identity())
    print("The extended Hadamard gate using matrix representation: ")
    print(q0_Hadamard.real, "\n")

    print("The bell state prior to the application of HADAMARD gate: ")
    print(b00, "\n")

    print("The bell state after the application HADAMARD gate: ")
    H_b00 = q0_Hadamard @ b00
    print(H_b00, "\n")

    print("Measurements after application of CNOT(0, 1)-gate")
    print("-------------------------------------------------------------\n")
    psi, outcome, counts, meas_probs = measure(H_b00, 1000)
    print("State of the system after measurement: ")
    print(psi, "\n")
    print("Observed states during measurements: ")
    print(f"|00> : {counts[0]}")
    print(f"|10> : {counts[1]}")
    print(f"|01> : {counts[2]}")
    print(f"|11> : {counts[3]} \n")
    print("Observed probabilities for each of the basis states of the bell pair: ")
    print(f"|00> : {meas_probs[0]*100}%")
    print(f"|10> : {meas_probs[1]*100}%")
    print(f"|01> : {meas_probs[2]*100}%")
    print(f"|11> : {meas_probs[3]*100}% \n\n")

    print(
        "Before proceeding with the application of both gates to the bell pair, we are going\n"
        "to check what would happen if we applied the Hadamard gate to the second qubit of the\n"
        "bell pair instead of the first.\n"
    )

    q1_Hadamard = np.kron(Identity(), HADAMARD)
    print("The extended Hadamard gate using matrix representation: ")
    print(q1_Hadamard.real, "\n")

    print("The bell state prior to the application of HADAMARD gate: ")
    print(b00, "\n")

    print("The bell state after the application HADAMARD gate: ")
    H_b00 = q1_Hadamard @ b00
    print(H_b00, "\n")

    print(
        "As we can see, the action of applying the Hadamard gate to the second qubit of the \n"
        "bell pair results in the exact same state as in the case of applying the Hadamard gate \n"
        "to the first qubit of the pair. We can now proceed with the application of CNOT(0, 1)-gate \n"
        "and thereafter proceeding with an application of the Hadamard gate to the first qubit. \n\n"
    )

    print("Applying the CNOT(0, 1)-gate and Hadamard gate to the bell state: ")
    print(
        "---------------------------------------------------------------------------------"
    )

    print("The bell state prior to the application of CNOT(0, 1) and HADAMARD gates: ")
    print(b00, "\n")

    CNOT_b00 = CNOT @ b00
    HCNOT_b00 = np.kron(Hadamard(), Identity()) @ CNOT_b00
    HCNOT_b00 = np.where(HCNOT_b00 > 1e-10, HCNOT_b00, 0)
    print("The bell state after the application CNOT(0, 1) and HADAMARD gates: ")
    print(HCNOT_b00)

    print(
        "What we did now, was to uncompute the entanglement. Thus, we are back to the 2-qubit system\n"
        "prior to the application of the entangler circuit which created the bell pair. We would achieve \n"
        "the exact same result if we computed the adjoints of the CNOT(0,1) and Hadamard gate, and \n"
        "applied them in the same order they are applied in the entangler circuit. Otherwise, we have \n"
        "to apply the gates in the opposite order. \n"
    )

    Hadamard_adj = np.kron(Hadamard(), Identity()).conj().T
    CNOT_adj = Cnot().conj().T
    print("Result of applying the adjoint of the gates to the b00 state: ")
    print(Hadamard_adj @ CNOT_adj @ b00, "\n")

    print("Measurements after application of CNOT(0, 1) and Hadamar gate to b00")
    print("-------------------------------------------------------------\n")
    psi, outcome, counts, meas_probs = measure(HCNOT_b00, 1000)
    print("State of the system after measurement: ")
    print(psi, "\n")
    print("Observed states during measurements: ")
    print(f"|00> : {counts[0]}")
    print(f"|10> : {0}")
    print(f"|01> : {0}")
    print(f"|11> : {0} \n")
    print("Observed probabilities for each of the basis states of the bell pair: ")
    print(f"|00> : {meas_probs[0]*100}%")
    print(f"|10> : {0.0}%")
    print(f"|01> : {0.0}%")
    print(f"|11> : {0.0}% \n\n")
