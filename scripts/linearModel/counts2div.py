import numpy as np
import pandas as pd
from pathlib import Path
from helpers import COUNTS_PATH, DATA_PATH

class Count_collection:

    def __init__(self, filepath:Path, metafile:Path, outpath:Path, save:bool = True):
        self.filepath = filepath
        self.metafile = metafile
        self.outpath = outpath
        self.df_meta = pd.read_csv(
            self.metafile, sep='\t',
            index_col="SCount"
        )
        self.counts = []
        self.df_diversity = None
        self._create_df_div(save)

    def _create_df_div(self, save):
        files = list(Path.glob(self.filepath, "*"))
        rows = []
        not_found = []
        for f in files:
            #FIXME: some sample_names not found in metafile --> naming of samples?
            try:
                print(f)
                s_name = f.name.split('_')[0]
                t = self.df_meta.loc[s_name, "Infektionsdauer[d]"]
                w = np.nan    #FIXME: Add weights
                count = Count(f, t, w)
                row = [s_name, t, w]
                row.extend(count.diversity.tolist())
                rows.append(row)
            except:
                s_name = f.name.split('_')[0]
                not_found.append(s_name)

        # print all samples that were not found in metafile
        from pprint import pprint
        print(f"\nSample names not found: {len(set(not_found))}")
        pprint(not_found)
        print('\n')

        colnames = ["name", "time", "sample_weight"]
        colnames.extend(count.count_df.index)
        self.df_diversity = pd.DataFrame(rows, columns=colnames)

        if save:
            outfile = self.outpath / "diversities.tsv"
            self.df_diversity.to_csv(outfile, sep='\t', header=True)



    def _create_div_matrix(self):
        pass


class Count:

    def __init__(self, file: Path, time=None, weight=None):
        self.file = file
        self.count_df = pd.read_csv(
            self.file, sep='\t',
            usecols=["pos1", "A", "C", "G", "T"],
            index_col="pos1"
        )
        self.count_df.rename('site_{}'.format, inplace=True)
        self.s_name = self.file.name.split('_')[0]
        self.diversity = None
        self.time = time
        self.weight = weight
        self.calc_diversity()


    def calc_diversity(self):
        coverage = self.count_df.apply(sum, axis=1)
        counts_freq = self.count_df.apply(lambda x: x/coverage, axis=0)
        self.diversity = 1 - np.power(counts_freq, 2).sum(axis=1, min_count=1)



PATH_META = Path("~/work/rki/HIV_Project/data/metadata/metadata.csv")
count = Count(Path("~/work/rki/HIV_Project/data/counts/00-00030_SEPE.counts.tsv"))
count.calc_diversity()


c = Count_collection(COUNTS_PATH, PATH_META, DATA_PATH)
