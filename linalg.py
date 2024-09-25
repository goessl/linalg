import numpy as np
from functools import reduce
from operator import mul
from fractions import Fraction



def randf(shape=None, precision=1000):
    """Return a random `Fraction` or an array of random `Fraction`s.
    
    Their numerators `n` are `-precision <= n <= +precision`
    and their denominators `d` are `1 <= d <= +precision`.
    """
    n = np.random.randint(-precision, +precision+1, shape)
    d = np.random.randint(1, +precision+1, shape)
    return Fraction(n, d) if shape is None else np.vectorize(Fraction)(n, d)


def _prod(iterable):
    """Like `math.prod` but for non-numeric types.
    
    `math.prod` might reject non-numeric types:
    https://docs.python.org/3/library/math.html#math.prod.
    For `float`s keep using `math.prod` for better precision.
    """
    return reduce(mul, iterable)


def submatrix(A, i, j):
    """Return a copy of `A` without the `i`-th row and `j`-th column."""
    return np.delete(np.delete(A, i, 0), j, 1)





# - - - Leibniz - - -
def _permutations(iterable, r=None):
    """`itertools.permutation`, but yields `permutation, parity`."""
    #https://docs.python.org/3/library/itertools.html#itertools.permutations
    #https://stackoverflow.com/a/69210050
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    
    indices = list(range(n))
    cycles = list(range(n, n-r, -1))
    parity = 0
    yield tuple(pool[i] for i in indices[:r]), +1
    
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                if ((n-i) & 1) == 0:
                    parity = 1 - parity
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                parity = 1 - parity
                yield tuple(pool[i] for i in indices[:r]), -2*parity+1
                break
        else:
            return

def minor_leibniz(A, i, j):
    """Return the `i,j`-th minor of `A`.
    
    See `det_leibniz` for more.
    """
    return det_leibniz(submatrix(A, i, j))

def det_leibniz(A):
    """Return the determinant of `A`.
    
    Calculates the determinant by the Leibniz formula.
    """
    return sum(p * _prod(A[tuple(range(len(s))), s]) \
            for s, p in _permutations(range(A.shape[0])))


if __name__ == '__main__':
    for _ in range(100):
        N = np.random.randint(1, 7)
        A = randf((N, N))
        
        prediction = det_leibniz(A)
        actual = np.linalg.det(A.astype(np.float64))
        assert np.isclose(float(prediction), actual)





# - - - Laplace - - -
def minor_laplace(A, i, j):
    """Return the `i,j`-th minor of `A`.
    
    See `det_laplace` for more.
    """
    return det_laplace(submatrix(A, i, j))

def det_laplace(A):
    """Return the determinant of `A`.
    
    Calculates the determinant by Laplace expansion.
    Uses the row/column with the most zero elements.
    """
    #https://en.wikipedia.org/wiki/Laplace_expansion
    if A.shape == (0, 0):
        return 1
    elif A.shape == (1, 1):
        return A[0,0]
    elif A.shape == (2, 2):
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]
    
    nonzeros_in_rows = np.count_nonzero(A, axis=1)
    nonzeros_in_columns = np.count_nonzero(A, axis=0)
    if min(nonzeros_in_rows) <= min(nonzeros_in_columns):
        i = np.argmin(nonzeros_in_rows)
        return sum((-1)**(i+j)*aij*minor_laplace(A, i, j) \
                for j, aij in enumerate(A[i,:]) if aij)
    else:
        j = np.argmin(nonzeros_in_columns)
        return sum((-1)**(i+j)*aij*minor_laplace(A, i, j) \
                for i, aij in enumerate(A[:,j]) if aij)


if __name__ == '__main__':
    for _ in range(100):
        N = np.random.randint(1, 6)
        A = randf((N, N))
        
        prediction = det_laplace(A)
        actual = np.linalg.det(A.astype(np.float64))
        assert np.isclose(float(prediction), actual)
