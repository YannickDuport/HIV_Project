import numpy as np
import pandas as pd
from pathlib import Path


class MSA_collection:

    def __init__(self, filepath, save=False):
        self.filepath = filepath
        self._create_MSA(save)

    def _create_MSA(self, save):
        msas = []
        data = []
        files = list(Path.glob(self.filepath, "*.fasta"))
        files.sort()
        for f in files:
            msa = MSA(f)
            msas.append(msa)

            name = msa.filepath.stem
            time = msa.time
            div = msa.diversity
            # weight = 1
            weight = np.random.choice([0.5, 1, 20])
            data.append([name, time, weight, div]) # sample weight currently hard-coded at 1 (no sample weights)

        self.msa_collection = np.array(msas)
        self.info = pd.DataFrame(data, columns=['name', 'time', "sample_weight", 'diversity_per_site'])

        if save:
            savepath = self.filepath / "diversities.csv"
            print(f"saving diversities under '{savepath}'")

            data_save = [l[:-1] + l[-1] for l in data]
            site_names = [f"div_site_{i}" for i in range(len(self.info.diversity_per_site[0]))]
            df_save = pd.DataFrame(data_save, columns=['name', 'time', 'sample_weight']+site_names)
            df_save.to_csv(savepath, header=True, index=False)

class MSA:

    def __init__(self, filepath):
        self.filepath = filepath
        self.msa = self._read_file()
        self.diversity = self.diversity_per_site()
        self.time = int(self.filepath.stem.split('_')[-1][1:])

    def _read_file(self):
        msa = []
        with open(self.filepath, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    continue
                msa.append(line[:-1])
        return np.array([list(string) for string in msa])

    def diversity_per_site(self):
        cov = len(self.msa)
        diversities = []
        for site in self.msa.transpose():
            base, count = np.unique(site, return_counts=True)
            diversities.append(self.calc_diversity(cov, count))
        return diversities

    def calc_diversity(self, coverage, counts):
        """ Calculate Nucleotide diversity (pi) per site for one timepoint
        Calculates the expected nucleotide diversity (E(D)) at each site, according to Zhao, Illingworth 2019.
        E(D) = 1 - sum(p_a^2), where p_a is the relative frequency of the symbol 'a' (A, C, G, T) at a given site.
        """
        # calculate p_a for A, C, G, T
        counts_freq = counts / coverage

        # calculate expected nucleotide diversity
        pi = 1 - np.power(counts_freq, 2).sum(axis=0)

        # replace NaN with 0
        np.nan_to_num(pi, 0)
        return pi
