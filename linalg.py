import numpy as np
from functools import reduce
from operator import mul
from fractions import Fraction
from tqdm.auto import tqdm



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


def assert_matrix(A):
    """Assert matrix."""
    assert A.ndim == 2

def assert_sqmatrix(A):
    """Assert square matrix."""
    assert A.ndim==2 and A.shape[0]==A.shape[1]


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





# - - - progress - - -
class Progress:
    """Progress handler for algorithms.
    
    Use as context. Total number of operations must be known beforehand.
    Call `update(op, n=1)` to increment tracking.
    """
    
    def __init__(self, totals, descprefix=''):
        """Create a new progress handler.
        
        `totals` should be a dictionary with the tracked operations as keys
        and the total number of operations as values.
        """
        self.pbars = \
                {o:tqdm(desc=descprefix+o, total=t) for o, t in totals.items()}
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        for pbar in self.pbars.values():
            pbar.close()
    
    def update(self, op, n=1):
        """Increment the operation `op` progress by `n`.
        
        If `op` is not tracked nothing happens.
        """
        if op in self.pbars:
            self.pbars[op].update(n)
    
    def add(self, a, b):
        c = a + b
        self.update('+')
        return c
    
    def mul(self, a, b):
        c = a * b
        self.update('*')
        return c
    
    def sum(self, iterable):
        return reduce(self.add, iterable)
    
    def prod(self, iterable):
        return reduce(self.mul, iterable)





# - - - matmul - - -
def matmul(A, B, progress=set()):
    """Return the matrix product of `A` & `B`.
    
    Matrices must be non-empty (L,N,M>0).
    For a LxN- & a NxM-matrix (where the result will be LxM) there will be
    - L(N-1)M additions (`+`),
    - LNM multiplications (`*`),
    - so L(2N-1)M operations in total.
    """
    assert_matrix(A)
    assert_matrix(B)
    assert A.shape[1] == B.shape[0]
    totals = {
        '+': A.shape[0] * (A.shape[1] - 1) * B.shape[1],
        '*': A.shape[0] * A.shape[1] * B.shape[1]
    }
    with Progress({o:totals[o] for o in progress}, 'matmul ') as progress:
        C = np.empty((A.shape[0], B.shape[1]), dtype=np.result_type(A, B))
        for i, j in np.ndindex(*C.shape):
            C[i, j] = progress.sum(progress.mul(aik, bkj)
                    for aik, bkj in zip(A[i, :],  B[:, j]))
        return C


if __name__ == '__main__':
    for _ in range(100):
        L, N, M = randint(1, 20, size=3)
        A, B = randf((L, N)), randf((N, M))
        
        prediction = matmul(A, B)
        actual = A @ B
        assert np.array_equal(prediction, actual)





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

def det_leibniz(A):
    """Return the determinant of `A`.
    
    Calculates the determinant by the Leibniz formula.
    """
    assert_sqmatrix(A)
    return sum(p * _prod(A[tuple(range(len(s))), s]) \
            for s, p in _permutations(range(A.shape[0])))


if __name__ == '__main__':
    for _ in range(100):
        N = randint(1, 7)
        A = randf((N, N))
        
        prediction = det_leibniz(A)
        actual = np.linalg.det(A.astype(np.float64))
        assert np.isclose(float(prediction), actual)





# - - - Gauss - - -
def det_gauss(A):
    """Return the determinant of `A`.
    
    Calculates the determinant by Gaussian elimination with complete pivoting.
    The matrix will be transformed in place into an upper triangular matrix
    (columns left of pivot won't be reduced).
    """
    #https://en.wikipedia.org/wiki/Gaussian_elimination#Computing_determinants
    assert_sqmatrix(A)
    s = True
    for i in range(A.shape[0]):
        #pivot
        i_max, j_max = \
                np.unravel_index(np.argmax(abs(A[i:, i:])), A[i:, i:].shape)
        if not A[i+i_max, i+j_max]:
            return A[i+i_max, i+j_max]
        swap_pivot(A, i, i+i_max, i+j_max)
        s ^= bool(i_max) != bool(j_max)
        #reduce (not left of pivot, these elements will not influence result)
        A[i+1:, i:] -= A[i, i:] * (A[i+1:, i] / A[i, i])[:, np.newaxis]
    
    return +_prod(np.diag(A)) if s else -_prod(np.diag(A))

def inv_gauss(A):
    """Return the inverse of `A`.
    
    Calculates the inverse by Gaussian elimination with complete pivoting.
    The matrix will be transformed in place into the identity matrix.
    """
    #https://math.stackexchange.com/a/744213/1170417
    assert_sqmatrix(A)
    P, Q = np.eye(A.shape[0], dtype=A.dtype), np.eye(A.shape[0], dtype=A.dtype)
    for i in range(A.shape[0]):
        #pivot
        i_max, j_max = \
                np.unravel_index(np.argmax(abs(A[i:, i:])), A[i:, i:].shape)
        if not A[i+i_max, i+j_max]:
            raise ZeroDivisionError
        swap_pivot(A, i, i+i_max, i+j_max)
        swap_rows(P, i, i+i_max), swap_columns(Q, i, i+j_max)
        #normalize pivot
        P[i, :] /= A[i, i]
        A[i, :] /= A[i, i]
        #zeros above and below
        P[:i, :] -= P[i, :] * A[:i, i][:, np.newaxis]
        A[:i, :] -= A[i, :] * A[:i, i][:, np.newaxis]
        P[i+1:, :] -= P[i, :] * A[i+1:, i][:, np.newaxis]
        A[i+1:, :] -= A[i, :] * A[i+1:, i][:, np.newaxis]
    return Q @ P


if __name__ == '__main__':
    for _ in range(100):
        N = randint(1, 10)
        A = randf((N, N))
        
        prediction = det_gauss(A.copy())
        actual = np.linalg.det(A.astype(np.float64))
        assert np.isclose(float(prediction), actual)
        
        A_inv = inv_gauss(A.copy())
        assert np.array_equal(A@A_inv, np.eye(N))





# - - - Bareiss - - -
def det_bareiss(A):
    """Return the determinant of an integer matrix `A`.
    
    Calculates the determinant by the Bareiss algorithm.
    Transforms `A` in place.
    """
    #https://en.wikipedia.org/wiki/Bareiss_algorithm#The_algorithm
    assert_sqmatrix(A)
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





#Gauss above because of default argument
#Laplace below because of usage
def minor(A, i, j, det=det_gauss):
    """Return the `i,j`-th minor of `A`.
    
    Uses the given determinant algorithm.
    """
    return det(submatrix(A, i, j))

def cofactor(A, i, j, det=det_gauss):
    """Return the `i,j`-th cofactor of `A`.
    
    Uses the given determinant algorithm.
    """
    return (-1)**(i+j)*minor(A, i, j, det)

def adj(A, det=det_gauss):
    """Return the adjugate of `A`.
    
    Uses the given determinant algorithm.
    """
    return cof(A, det).T

def cof(A, det=det_gauss):
    """Return the cofactor matrix of `A`.
    
    Uses the given determinant algorithm.
    """
    cofA = np.empty_like(A)
    for i in np.ndindex(*cofA.shape):
        cofA[*i] = cofactor(A, *i, det)
    return cofA

if __name__ == '__main__':
    for _ in range(100):
        N = randint(2, 6)
        A = randf((N, N))
        
        adjA = adj(A)
        assert np.array_equal(A@adjA, det_gauss(A)*np.eye(N, dtype=int))





# - - - Laplace - - -
def det_laplace(A):
    """Return the determinant of `A`.
    
    Calculates the determinant by Laplace expansion.
    Uses the row/column with the most zero elements.
    """
    #https://en.wikipedia.org/wiki/Laplace_expansion
    assert_sqmatrix(A)
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
        return sum(aij*cofactor(A, i, j, det_laplace) \
                for j, aij in enumerate(A[i,:]) if aij)
    else:
        j = np.argmin(nonzeros_in_columns)
        return sum(aij*cofactor(A, i, j, det_laplace) \
                for i, aij in enumerate(A[:,j]) if aij)


if __name__ == '__main__':
    for _ in range(100):
        N = randint(1, 6)
        A = randf((N, N))
        
        prediction = det_laplace(A)
        actual = np.linalg.det(A.astype(np.float64))
        assert np.isclose(float(prediction), actual)
