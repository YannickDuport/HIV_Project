import numpy as np

from pathlib import Path
from sklearn.linear_model import LinearRegression
from scripts.linearModel.helpers import count_lines


class Count:
    def __init__(self, filepath: Path):
        """Calculates nucleotide diversities per site and performs a linear regression
        Recieves a filepath containing count files (count_*.tsv) for each timepoint and a file containing all timepoints (time_*.tsv)
        Calculates nucleotide diversity (pi) per site for each count file and stores them in a 2D-ndarray (i - timepoint, j - site)

        From this array a linear-regression model is being fit on the data:
            dt_j = sum(pi_ij * w_i)

        :param filepath:
        """
        self.filepath = filepath
        self.time = self._read_time()
        self.divergence = self._calculate_pi()
        self.omega = None
        self.intercept = None

    def _read_time(self):
        file = list(self.filepath.glob("time*.tsv"))
        if len(file) != 1:
            print(self.filepath)
            print(file)
            raise ValueError("Either no or too many files containing timepoints in the given directory")
        time = np.genfromtxt(file[0], delimiter='\t')
        return time

    def _pi_per_tp(self, count_array):
        """ Calculate Nucleotide diversity (pi) per site for one timepoint
        Calculates the expected nucleotide diversity (E(D)) at each site, according to Zhao, Illingworth 2019.
        E(D) = 1 - sum(p_a^2), where p_a is the relative frequency of the symbol 'a' (A, C, G, T) at a given site.
        """
        # calculate p_a for A, C, G, T
        cov = count_array.sum(axis=0)
        counts_freq = count_array / cov

        # calculate expected nucleotide diversity
        pi = 1 - np.power(counts_freq, 2).sum(axis=0)

        # replace NaN with 0
        np.nan_to_num(pi, 0)
        return pi

    def _calculate_pi(self):
        """Calculates Nucleotide (pi) diversity for each timepoint
        """
        # get filenames
        files = list(self.filepath.glob("counts*.tsv"))
        files.sort()
        if len(files) != len(self.time):
            raise ValueError("Number of count files doesn't match number of timepoints")

        # create empty 2D-ndarray
        n_rows = len(self.time)
        n_cols = count_lines(files[0])      # NOTE: All files should have the same length
        pi = np.zeros(shape=(n_rows, n_cols))

        # convert count files to ndarray and calculate pi
        for i, f in enumerate(files):
            count_array = np.genfromtxt(f, delimiter='\t', usecols=(1, 2, 3, 4)).transpose()
            pi[i] = self._pi_per_tp(count_array)

        return pi

    def linreg(self):
        print(self.divergence)
        reg = LinearRegression().fit(self.divergence, self.time)
        self.omega = reg.coef_
        self.intercept = reg.intercept_
        print(f"omega = {self.omega}")
        print(f"intercept = {self.intercept}")
