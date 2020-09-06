from pathlib import Path
import numpy as np
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

def calc_aic(x, y, coefficients):
    n = len(y)
    sigma = np.var(y)
    df = np.count_nonzero(coefficients, axis=0)

    y_2d = [[k] for k in y]
    y_hat = np.dot(x, coefficients)
    residuals = y_2d - y_hat
    mse = np.power(residuals, 2).mean(axis=0)
    aic = n*mse/sigma + 2*df
    return aic


def calc_aic_depr(n, p, mse):
    return n * np.log(np.mean(mse)) + 2 * p