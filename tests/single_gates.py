from __future__ import annotations
import numpy as np
import math
from src.solvers import *
from src.ops import *
from src.bell_states import *

if __name__ == "__main__":
    # Preparing two seperate qubits each in one of the basis state |0> and |1>
    psi0 = state(alpha=1.0)
    psi1 = state(beta=1.0)

    # number of measurements
    shots = 1000

    choices = ["X", "Y", "Z", "H", "S"]
    print("Enter one of the listed commands to apply a quantum gate: ")
    print(choices)
    command = str(input("Enter command: "))

    while command in choices:
        if command == "X":
            print("Applying PauliX gate to state [1.0j, 0.0j]")
            print("----------------------------------------------------")
            probs, counts, measurement_probs, post_psi = measurement(
                psi0, PauliX(), shots
            )

            print(
                f"State of the qubit prior to applying pauli X matrix: \
                        \n|0>: {psi0[0]}   |1>: {psi0[1]}\n"
            )
            print(
                f"State of the qubit after applying pauli X matrix: \
                        \n|0>: {post_psi[0]}   |1>: {post_psi[1]}\n"
            )
            print(f"*Performing {shots} measurements*")
            print("-------------------------------------")
            print(
                f"Probabilty after applying pauli X gate: \
                  \n|0> : {probs[0]*100}%  \
                  \n|1> : {probs[1]*100}% \n"
            )
            print(
                f"Observed states: \
                    \n|0> : {shots - counts[0]} \
                    \n|1> : {counts[0]}\n"
            )
            print(
                f"Observed probabilities: \
                    \n|0> : {100 - measurement_probs[0]*100}% \
                    \n|1> : {measurement_probs[0] * 100}%  "
            )
            print("\n")

            print("Applying PauliX gate to state [0.0j, 1.0j]")
            print("----------------------------------------------------")
            probs, counts, measurement_probs, post_psi = measurement(
                psi1, PauliX(), shots
            )

            print(
                f"State of the qubit prior to applying pauli X matrix: \
                        \n|0>: {psi1[0]}   |1>: {psi1[1]}\n"
            )
            print(
                f"State of the qubit after applying pauli X matrix: \
                        \n|0>: {post_psi[0]}   |1>: {post_psi[1]}\n"
            )
            print(f"*Performing {shots} measurements*")
            print("-------------------------------------")
            print(
                f"Probabilty after applying pauli X gate: \
                  \n|0> : {probs[0]*100}%  \
                  \n|1> : {probs[1]*100}% \n"
            )
            print(
                f"Observed states: \
                    \n|0> : {counts[0]}\
                    \n|1> : {shots - counts[0]} \n"
            )
            print(
                f"Observed probabilities: \
                    \n|0> : {measurement_probs[0] * 100}%  \
                    \n|1> : {100 - measurement_probs[0]*100}% \n"
            )
            print("----------------------------------------------------\n")

        elif command == "Y":
            print("Applying PauliY gate to state [1.0j, 0.0j]")
            print("----------------------------------------------------")
            probs, counts, measurement_probs, post_psi2 = measurement(
                psi0, PauliY(), shots
            )
            print(
                f"State of the qubit prior to applying pauli Y matrix: \
                        \n|0>: {psi0[0]}   |1>: {psi0[1]}\n"
            )
            print(
                f"State of the qubit after applying pauli Y matrix: \
                        \n|0>: {post_psi2[0]}   |1>: {post_psi2[1]}\n"
            )
            print(f"*Performing {shots} measurements*")
            print("-------------------------------------")
            print(
                f"Probabilty after applying pauli Y gate: \
                  \n|0> : {probs[0]*100}%  \
                  \n|1> : {probs[1]*100}% \n"
            )
            print(
                f"Observed states: \
                    \n|0> : {shots - counts[0]} \
                    \n|1> : {counts[0]}\n"
            )
            print(
                f"Observed probabilities: \
                    \n|0> : {100 - measurement_probs[0]*100}% \
                    \n|1> : {measurement_probs[0] * 100}%  \n"
            )
            print("\n")

            print("Applying PauliY gate to state [0.0j, 1.0j]")
            print("----------------------------------------------------")
            probs, counts, measurement_probs, post_psi2 = measurement(
                psi1, PauliY(), shots
            )
            print(
                f"State of the qubit prior to applying pauli Y matrix: \
                        \n|0>: {psi1[0]}   |1>: {psi1[1]}\n"
            )
            print(
                f"State of the qubit after applying pauli Y matrix: \
                        \n|0>: {post_psi2[0]}   |1>: {post_psi2[1]}\n"
            )
            print(f"*Performing {shots} measurements*")
            print("-------------------------------------")
            print(
                f"Probabilty after applying pauli Y gate: \
                  \n|0> : {probs[0]*100}%  \
                  \n|1> : {probs[1]*100}% \n"
            )
            print(
                f"Observed states: \
                    \n|0> : {counts[0]}\
                    \n|1> : {shots - counts[0]} \n"
            )
            print(
                f"Observed probabilities: \
                    \n|0> : {measurement_probs[0] * 100}%  \
                    \n|1> : {100 - measurement_probs[0]*100}% "
            )
            print("----------------------------------------------------\n")

        elif command == "Z":
            print("Applying PauliZ gate to state [1.0j, 0.0j]")
            print("----------------------------------------------------")
            probs, counts, measurement_probs, post_psi2 = measurement(
                psi0, PauliZ(), shots
            )
            print(
                f"State of the qubit prior to applying pauli Z matrix: \
                        \n|0>: {psi0[0]}   |1>: {psi0[1]}\n"
            )
            print(
                f"State of the qubit after applying pauli Z matrix: \
                        \n|0>: {post_psi2[0]}   |1>: {post_psi2[1]}\n"
            )
            print(f"*Performing {shots} measurements*")
            print("-------------------------------------")
            print(
                f"Probabilty after applying pauli Z gate: \
                  \n|0> : {probs[0]*100}%  \
                  \n|1> : {probs[1]*100}% \n"
            )
            print(
                f"Observed states: \
                    \n|0> : {counts[0]}\
                    \n|1> : {shots - counts[0]} \n"
            )
            print(
                f"Observed probabilities: \
                    \n|0> : {measurement_probs[0] * 100}% \
                    \n|1> : {100 - measurement_probs[0]*100}% \n"
            )
            print("\n")

            print("Applying PauliZ gate to state [0.0j, 1.0j]")
            print("----------------------------------------------------")
            probs, counts, measurement_probs, post_psi2 = measurement(
                psi1, PauliZ(), shots
            )
            print(
                f"State of the qubit prior to applying pauli Z matrix: \
                        \n|0>: {psi1[0]}   |1>: {psi1[1]}\n"
            )
            print(
                f"State of the qubit after applying pauli Z matrix: \
                        \n|0>: {post_psi2[0]}   |1>: {post_psi2[1]}\n"
            )
            print(f"*Performing {shots} measurements*")
            print("-------------------------------------")
            print(
                f"Probabilty after applying pauli Z gate: \
                  \n|0> : {probs[0]*100}%  \
                  \n|1> : {probs[1]*100}% \n"
            )
            print(
                f"Observed states: \
                    \n|0> : {shots - counts[0]} \
                    \n|1> : {counts[0]}\n"
            )
            print(
                f"Observed probabilities: \
                    \n|0> : {100 - measurement_probs[0]*100}% \
                    \n|1> : {measurement_probs[0] * 100}%"
            )
            print("----------------------------------------------------\n")

        elif command == "H":
            print("Applying Hadamard gate to state [1.0j, 0.0j]")
            print("----------------------------------------------------")
            probs, counts, measurement_probs, post_psi2 = measurement(
                psi0, Hadamard(), shots
            )
            print(
                f"State of the qubit prior to applying Hadamard matrix: \
                        \n|0>: {psi0[0]}   |1>: {psi0[1]}\n"
            )
            print(
                f"State of the qubit after applying Hadamard matrix: \
                        \n|0>: {post_psi2[0]}   |1>: {post_psi2[1]}\n"
            )
            print(f"*Performing {shots} measurements*")
            print("-------------------------------------")
            print(
                f"Probabilty after applying Hadamard gate: \
                  \n|0> : {probs[0]*100}%  \
                  \n|1> : {probs[1]*100}% \n"
            )
            print(
                f"Observed states: \
                    \n|0> : {counts[0]} \
                    \n|1> : {counts[1]}\n"
            )
            print(
                f"Observed probabilities: \
                    \n|0> : {measurement_probs[0] * 100}% \
                    \n|1> : {measurement_probs[1] * 100}%"
            )
            print("\n")

            print("Applying Hadamard gate to state [0.0j, 1.0j]")
            print("----------------------------------------------------")
            probs, counts, measurement_probs, post_psi2 = measurement(
                psi1, Hadamard(), shots
            )
            print(
                f"State of the qubit prior to applying Hadamard matrix: \
                        \n|0>: {psi1[0]}   |1>: {psi1[1]}\n"
            )
            print(
                f"State of the qubit after applying Hadamard matrix: \
                        \n|0>: {post_psi2[0]}   |1>: {post_psi2[1]}\n"
            )
            print(f"*Performing {shots} measurements*")
            print("-------------------------------------")
            print(
                f"Probabilty after applying Hadamard gate: \
                  \n|0> : {probs[0]*100}%  \
                  \n|1> : {probs[1]*100}% \n"
            )
            print(
                f"Observed states: \
                    \n|0> : {counts[0]} \
                    \n|1> : {counts[1]}\n"
            )
            print(
                f"Observed probabilities: \
                    \n|0> : {measurement_probs[0] * 100}% \
                    \n|1> : {measurement_probs[1] * 100}%"
            )
            print("----------------------------------------------------\n")

        elif command == "S":
            print("Applying Phase/S- gate to state [1.0j, 0.0j]")
            print("----------------------------------------------------")
            probs, counts, measurement_probs, post_psi2 = measurement(
                psi0, Sgate(), shots
            )
            print(
                f"State of the qubit prior to applying phase matrix: \
                        \n|0>: {psi0[0]}   |1>: {psi0[1]}\n"
            )
            print(
                f"State of the qubit after applying phase matrix: \
                        \n|0>: {post_psi2[0]}   |1>: {post_psi2[1]}\n"
            )
            print(f"*Performing {shots} measurements*")
            print("-------------------------------------")
            print(
                f"Probabilty after applying Hadamard gate: \
                  \n|0> : {probs[0]*100}%  \
                  \n|1> : {probs[1]*100}% \n"
            )
            print(
                f"Observed states: \
                    \n|0> : {counts[0]}\
                    \n|1> : {shots - counts[0]} \n"
            )
            print(
                f"Observed probabilities: \
                    \n|0> : {measurement_probs[0] * 100}%  \
                    \n|1> : {100 - measurement_probs[0]*100}% "
            )
            print("\n")

            print("Applying Phase/S- gate to state [0.0j, 1.0j]")
            print("----------------------------------------------------")
            probs, counts, measurement_probs, post_psi2 = measurement(
                psi1, Sgate(), shots
            )
            print(
                f"State of the qubit prior to applying phase matrix: \
                        \n|0>: {psi1[0]}   |1>: {psi1[1]}\n"
            )
            print(
                f"State of the qubit after applying phase matrix: \
                        \n|0>: {post_psi2[0]}   |1>: {post_psi2[1]}\n"
            )
            print(f"*Performing {shots} measurements*")
            print("-------------------------------------")
            print(
                f"Probabilty after applying Hadamard gate: \
                  \n|0> : {probs[0]*100}%  \
                  \n|1> : {probs[1]*100}% \n"
            )
            print(
                f"Observed states: \
                    \n|0> : {shots - counts[0]} \
                    \n|1> : {counts[0]}\n"
            )
            print(
                f"Observed probabilities: \
                    \n|0> : {100 - measurement_probs[0]*100}% \
                    \n|1> : {measurement_probs[0] * 100}%  \n"
            )
            print("----------------------------------------------------\n")
        else:
            break

        print("Enter one of the listed commands to apply a quantum gate: ")
        print(choices)
        command = str(input("Enter command: "))
        print("\n")
