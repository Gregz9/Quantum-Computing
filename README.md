# Quantum-Computing
This repo contains code and report developed as part of a project in subject FYS5419 - Quantum Computing and Quantum Machine Learning by the University of Oslo. The main topic of the 
project is concerned about the simulation of one-, two- and many-body quantum systems, including the famous Lipkin model. All code used to solve the Hamiltonians of the different systems 
studied as part of this project can be found in directory "src", while the code used to generate the results presented in the report ("paper.pdf" in doc directory) can be found in a seperate 
directory within "src" called "runs". The latter contains six programs which correspond to the various parts of the the project, starting with the programs "gates_task_a.py" amd "bell_task_a.py", 
which correspond to part A. To run all the programs, including those two, you only need to run the following command: 

```
python3 <file_name.py>
```

Following those two files, are the scripts called "vqe_1qubit.py" and "vqe_2qubits.py" which can be used to solve the hamiltonian of the one- and two-qubit systems presented in part b thorugh e og the project, respectively. Everything within these scripts is adjusted, thus the only requirement for repreducing the results presented in the report, is to run the files using the "python3" command presented above followeed by the filename with ".py" extension. Finally the two files corresponding to results achieved for the the lipkin model are named using a convention which indicates which case we're studyingthe names of the scripts are written with "J1" adn J2" to indicate simulation of the lipkin model for systems with total spin equal to J=1 and J=2. 

Side note: Even though its not part of the report, we have added the possibility for those interested to simulate the full lipkin model, where the spin-exchange term W is not set to zero. This is only added for the Lipkin model with spin J=2, and requires the user to pass in extra parameters to the function "lipkin_H_J2_Pauli" in addtion to v. In order to perform the simulation you will have to add w values, and a boolean value "full" , and set it to true. The exact same thing has to be done for the "VQE_scipy" method. To make it slighlty more intuitive, the required parameters have been commented out, thus it shall be easy to simualte the full lipkin model.
