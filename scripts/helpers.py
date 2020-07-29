from pathlib import Path
import time

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
RESULT_PATH = BASE_PATH / "results"

def timeit(method):
    """Timing decorator"""

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('{:20}  {:8.4f} [s]'.format(method.__name__, (te - ts)))
        return result

    return timed


def count_lines(filename):
    """Returns the number of lines in a given file"""
    with open(filename) as f:
        return sum(1 for line in f)