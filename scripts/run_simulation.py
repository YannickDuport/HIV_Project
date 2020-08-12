import numpy as np

from pathlib import Path

from scripts.helpers import RESULT_PATH, DATA_PATH
from scripts.simulation import Simulation


result_path = RESULT_PATH / "simulation"
size = 100
length = 1000
p_mut = 1E-4
p_repl = 0.5
stable_pop_size = True
n_evolutions = 100
n_simulations = 30

data_path = DATA_PATH / "fasta"

for i in range(n_simulations):
    print(f"Running simulation {i+1} of {n_simulations}")
    infix = f"{i}_"
    initial_seq = np.random.choice([0, 1, 2, 3], length)
    sim = Simulation(size=size, length=length, p_mut=p_mut, p_repl=p_repl, stable_pop_size=stable_pop_size, initial_seq=initial_seq)
    for _ in range(n_evolutions):
        sim.evolution(result_path, infix)

    # select files for fitting
    files = list(Path.glob(result_path, f'*_{infix}*.fasta'))
    files_copy = np.random.choice(files, size=5, replace=False)
    for f in files_copy:
        f.rename(data_path / f.name)


