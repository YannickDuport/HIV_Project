import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

from scripts.helpers import RESULT_PATH, DATA_PATH
from scripts.simulation import Simulation
from scripts.msa import MSA
from scripts.msa import MSA_collection

result_path = RESULT_PATH / "simulation"
size = 100
length = 1000
p_mut = 1E-4
p_repl = 0.5
stable_pop_size = True
n_evolutions = 100
n_simulations = 10

data_path = DATA_PATH / "fasta"

for i in range(n_simulations):
    print(f"Running simulation {i+1} of {n_simulations}")
    infix = f"{i}_"
    initial_seq = np.random.choice([0, 1, 2, 3], length)
    sim = Simulation(size=size, length=length, p_mut=p_mut, p_repl=p_repl, stable_pop_size=stable_pop_size, initial_seq=initial_seq)
    for _ in range(n_evolutions):
        sim.evolution(result_path, infix)
    files = list(Path.glob(result_path, f'*_{infix}*.fasta'))
    files_copy = np.random.choice(files, size=15, replace=False)
    files_copy_test = np.random.choice(files_copy, size=3, replace=False)
    files_copy_train = np.array([f for f in files_copy if f not in files_copy_test])
    for f in files_copy_train:
        f.rename(data_path / "training" / f.name)
    for f in files_copy_test:
        f.rename(data_path / "testing" / f.name)

