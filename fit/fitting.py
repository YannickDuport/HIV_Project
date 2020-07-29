import numpy as np
from pathlib import Path
from scripts.count import Count

DATA_PATH = Path(__file__).parent.parent / "data" / "counts"

test = Count(DATA_PATH)
test.linreg()