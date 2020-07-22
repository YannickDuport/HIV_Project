import numpy as np
import pysam

class SAM:
    def __init__(self, filepath):
        self.filepath = filepath
        self.samfile = pysam.AlignmentFile(self.filepath, "rb")

    def count(self):
        for pileupcolumn in self.samfile.pileup():
            position = pileupcolumn.pos
            coverage = pileupcolumn.n
            for pileupread in pileupcolumn.pileups:
                if not pileupread.is_del and not pileupread.is_refskip:
                    base = pileupread.alignment.query_sequence[pileupread.query_position]