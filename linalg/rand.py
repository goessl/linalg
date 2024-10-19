import numpy as np
from fractions import Fraction



def randint(low, high=None, size=None):
    """Return a random `int` or an array of random `int`s.
    
    Like `numpy.random.randint` but with native `int`s.
    """
    if size is None:
        return int(np.random.randint(low, high))
    else:
        return np.random.randint(low, high, size=size).astype(object)

def randf(shape=None, precision=1000):
    """Return a random `Fraction` or an array of random `Fraction`s.
    
    Their numerators `n` are `-precision <= n <= +precision`
    and their denominators `d` are `1 <= d <= +precision`.
    """
    n = randint(-precision, +precision+1, shape)
    d = randint(1, +precision+1, shape)
    return Fraction(n, d) if shape is None else np.vectorize(Fraction)(n, d)

def randfr(M, N, r, precision=1000):
    """Return a random MxN `Fraction` matrix of rank `r`."""
    #https://math.stackexchange.com/a/757864/1170417
    U, V = randf((M, M), precision), randf((N, N), precision)
    P = np.zeros((M, N), dtype=np.int64)
    P[np.diag_indices(r)] = 1
    return U @ P @ V
