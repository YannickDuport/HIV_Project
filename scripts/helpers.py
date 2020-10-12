import numpy as np
import time

from itertools import chain, combinations
from pathlib import Path

from sklearn.metrics import mean_squared_error


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

def calc_aic_1d(x, y, coefficients, weights):
    n = len(y)
    sigma = np.var(y)
    df = np.count_nonzero(coefficients, axis=0)

    y_hat = np.dot(x, coefficients)
    mse = mean_squared_error(y, y_hat, sample_weight=weights)
    aic = n*mse/sigma + 2*df
    return aic


def calc_aic_depr(x, y, coefficients):
    n = len(y)
    df = np.count_nonzero(coefficients, axis=0)

    y_2d = [[k] for k in y]
    y_hat = np.dot(x, coefficients)
    residuals = y_2d - y_hat

    print(df)
    print(n*np.log(np.power(residuals, 2).sum(axis=0)))
    return n * np.log(np.power(residuals, 2).sum(axis=0)) + (2*np.power(df, 2) + 2*df) / (n - df -1)
    # return n * np.log(np.power(residuals, 2).sum(axis=0)) + 2 * df


def powerset(iterable, size):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, size))

def split(lst, n, random_state):
    """ Splits lst randomly into n chunks of approx. equal size"""

    # set seed for reproducability
    if random_state is not None:
        np.random.seed(random_state)

    # shuffle indices to introduce randomness
    idx = list(range(len(lst)))
    np.random.shuffle(idx)

    # split into n subsets
    k, m = divmod(len(idx), n)
    idx_chunks = [idx[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    return idx_chunks
    # return lst[[idx_chunks]]