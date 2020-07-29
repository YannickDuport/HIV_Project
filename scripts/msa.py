import numpy as np
from pathlib import Path


class MSA_collection:

    def __init__(self, filepath):
        self.filepath = filepath
        self.msa_collection = self._create_MSA()

    def _create_MSA(self):
        msas = []
        files = list(Path.glob(self.filepath, "*.fasta"))
        files.sort()
        for f in files:
            msas.append(MSA(f))
        return np.array(msas)


class MSA:

    def __init__(self, filepath):
        self.filepath = filepath
        self.msa = self._read_file()
        self.diversity = self.diversity_per_site()

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
