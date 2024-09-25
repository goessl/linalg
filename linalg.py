import numpy as np
from functools import reduce
from operator import mul
from fractions import Fraction



# - - - Random Creation - - -
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





# - - - Utility - - -
def _prod(iterable):
    """Like `math.prod` but for non-numeric types.
    
    `math.prod` might reject non-numeric types:
    https://docs.python.org/3/library/math.html#math.prod.
    For `float`s keep using `math.prod` for better precision.
    """
    return reduce(mul, iterable)


#https://stackoverflow.com/a/54069951
def swap_rows(A, i, j):
    """Swap the `i`-th and `j`-th row of `A` in-place."""
    A[[i, j], :] = A[[j, i], :]

def swap_columns(A, i, j):
    """Swap the `i`-th and `j`-th column of `A` in-place."""
    A[:, [i, j]] = A[:, [j, i]]

def swap_pivot(A, p, i, j):
    """Swap the `p`-&`i`-th rows and `p`-&`j`-th columns of `A` in-place."""
    swap_rows(A, p, i)
    swap_columns(A, p, j)


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
        N = randint(1, 7)
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
        N = randint(1, 6)
        A = randf((N, N))
        
        prediction = det_laplace(A)
        actual = np.linalg.det(A.astype(np.float64))
        assert np.isclose(float(prediction), actual)





# - - - Bareiss - - -
def det_bareiss(A):
    """Return the determinant of an integer matrix `A`.
    
    Calculates the determinant by the Bareiss algorithm.
    Transforms `A` in place.
    """
    #https://en.wikipedia.org/wiki/Bareiss_algorithm#The_algorithm
    s = True
    for i in range(A.shape[0]):
        #pivot
        i_max, j_max = \
                np.unravel_index(np.argmax(abs(A[i:, i:])), A[i:, i:].shape)
        #cancellation
        if not A[i+i_max, i+j_max]:
            return 0
        swap_pivot(A, i, i+i_max, i+j_max)
        s ^= bool(i_max) != bool(j_max)
        #reduction
        A[i+1:, i:] = \
                (A[i, i]*A[i+1:, i:] - A[i+1:, i][:, np.newaxis]*A[i, i:]) \
                // (A[i-1, i-1] if i>0 else 1)
    
    return +A[-1, -1] if s else -A[-1, -1]


if __name__ == '__main__':
    for _ in range(1000):
        N = randint(1, 10)
        A = randint(-100, +100, size=(N, N))
        
        prediction = det_bareiss(A.copy())
        actual = np.linalg.det(A.astype(np.int64))
        assert isinstance(prediction, int) \
                and np.isclose(float(prediction), actual)
