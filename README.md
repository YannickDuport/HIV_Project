# HIV_Project

## 1. Simulation
To validate the implemented tools we implemented a simulation of sequence evolution (replication and mutation). The 
simulation is based on a genetic algorithm.

### 1.1 Parameters
The following parameters can be defined in the simulation:

- size: population size (at t=0)
- length: sequence length 
- p_repl: replication rate. Determines the number of individuals chosen for replication. If stable_pop_size = True, the 
replication rate is being ignored and half the population (size/2) is chosen for replication each iteration.
- p_mut: mutation rate. Number of bases mutated is drawn from a poisson distribution centered around the mutation rate.
- initial_seq: Initial sequence in the population. The simulation assumes that at t=0 all individuals have the same sequence.
- stable_pop_size: Whether the population size can change over time. If stable_pop_size = True, the replication rate is 
being ignored and half the population (size/2) is chosen for replication each iteration.


## 2. General linear model
