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

### 1.2
Each iteration the following steps are executed:
1) Replication
2) Mutation 
3) Write Fasta

#### 1.2.1 Replication
During the replication step N individuals are chosen for replication.
The number of chosen individuals (N) is done the following way:

- If population size is set as stable:
  N = population size / 2
- If population size is set as unstable (size can change over time):
  N ~ Poisson(p_repl)

N individuals are then randomly chosen from the population. Each individual creates 2 offsprings, which inherit the same sequence as their parent.
Together, all offsprings form the population of the new population

#### 1.2.2 Mutation
Once the new population is formed, single bases are randomly selected for mutation.
The number of mutations (N_mut) is drawn from a poisson distribution centered around the mutation rate (p_mut)

N_mut ~ Poisson(p_mut)

Sites of mutations are then randomly chosen over all sites in all individuals.
Sites are mutated with a 60% chance for transitions (A -> T, T -> A, C -> G, G -> C) and a 40% chance for transversions.

#### 1.2.3 Write Fasta
Each iteration a MSA of the new population is written to a fasta file. 

### 1.3 Run Simulation
To run the simulation, an object of the class Simulation has to be initialized.

`sim = Simulation(size, length, p_repl, p_mut, initial_seq, stable_pop_size)`

Once initialized run 'evolution' progress the population by 1 timestep. The save path for the generated fasta-files can be defined with the argument 'results_path'.

`sim.evolution(results_path)`

## 2. General linear model
To fit the model to the data a linear model is used. Fitting can be done by lasso regression and elastic net regression.
The tool is built on [scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html).

### 2.1 Lasso regression
