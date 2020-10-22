import numpy as np


# FIXME: Case of dying population not being handled yet

class Simulation:
    def __init__(self, size, length, p_repl, p_mut, initial_seq=None, stable_pop_size=True):
        self.size = [size]
        self.length = length
        self.p_replication = p_repl
        self.p_mutation = p_mut
        self.stable_pop_size = stable_pop_size
        self.population = self.create_population(initial_seq)
        self.time = 0
        self.t_dict = {
            0: "A",
            1: "T",
            2: "C",
            3: "G"
        }

    def create_population(self, seq):
        if seq is None:
            return np.zeros(shape=(self.size, self.length))
        else:
            return np.repeat([seq], repeats=self.size, axis=0)

    def _replication(self):
        """ Selects individuals for replication

        If the population size is stable, a random subset of individuals are drawn from the population with size = popopulation_size / 2
        If the population size can change, the number of replication is sampled from a poisson distribution
        centered at replication_rate * population size

        :return: A list of (unmutated) offsprings (2 per selected individual)
        """
        # determine number of replicating individuals
        if self.stable_pop_size:
            n_replication = int(self.size[-1] / 2)
        else:
            n_replication = self.size[-1] + 1
            while n_replication > self.size[-1]:
                n_replication = np.random.poisson(self.p_replication * self.size[-1])

        # select replicating individuals with chosen method
        selections = np.random.choice(np.arange(0, len(self.population)), size=n_replication, replace=False)

        # create 2 offsprings from each selected individual
        new_population = np.repeat(self.population[selections], 2, axis=0)
        return new_population

    def _mutation(self, population):
        """ Introduces mutations into a population
        Number of new mutation are drawn from a poisson distribution centered at mutation_rate * number_of_sites. The
        sites that are mutated are determined randomly. 0 -> 1 and 1 -> 0 mutations have the same probability to occur.

        :param population:
        :return: mutated population
        """
        # transition / transversion probabilities
        alpha = 0.2     # transversion
        beta = 0.2      # transversion
        gamma = 0.6     # transition


        transitions = {
            "gamma": { # transition
                0: 1, 1: 0, 2: 3, 3: 2
            },
            "alpha":{ # transversion 1
                0: 2, 2: 0, 1: 3, 3: 1
            },
            "beta": { # transversion 2
                0: 3, 3: 0, 1: 2, 2: 1
            }
        }
        # draw number of new mutation from poisson distribution
        n_sites = self.length * (len(population))
        new_mut = np.random.poisson(n_sites * self.p_mutation)

        # determine mutation sites
        mut_sites = np.random.choice(np.arange(n_sites), new_mut, replace=False)
        row, col = np.divmod(mut_sites, self.length)

        for r, c in zip(row, col):
            tau = np.random.uniform()
            base = population[r][c]
            if tau < gamma:
                population[r][c] = transitions["gamma"][base]
            elif tau < gamma + alpha:
                population[r][c] = transitions["alpha"][base]
            elif tau <= 1:
                population[r][c] = transitions["beta"][base]

        return population

    def _update(self, population):
        """ Updates class values for a new population
        Updates the population and population_size

        :param population:
        :return:
        """
        # assign new population and population size
        self.population = population
        self.size.append(len(population))

        # change time
        self.time += 1

    def evolution(self, result_path, infix):
        """ Runs one iteration of evolution (replication, mutation)
        """
        parents = self._replication()
        new_pop = self._mutation(parents)
        self._update(new_pop)
        # print(f"population size: {self.size[-1]}")
        self._writeFasta(result_path, infix)

    def _writeFasta(self, result_path, infix=""):
        filename = f"population_{infix}{self.time:03}.fasta"
        filepath = result_path / filename
        i = 0
        with open(filepath, 'w+') as f:
            for line in self.population:
                f.write(f">alignment{self.time:03}_{i:05}\n")
                f.writelines("%s" % self.t_dict[int(char)] for char in line)
                f.write('\n')
                i += 1
        f.close()





