import numpy as np
from math import prod
from functools import reduce
from operator import mul


#https://stackoverflow.com/a/54069951
def swap_rows(A, i, j):
    A[[i, j], :] = A[[j, i], :]
def swap_columns(A, i, j):
    A[:, [i, j]] = A[:, [j, i]]
def swap_pivot(A, p, i, j):
    swap_rows(A, p, i)
    swap_columns(A, p, j)

def det_gauss(A, criterion=bool):
    """Return the determinant of A.
    
    Calculates the determinant of A by Gaussian elimination with complete pivoting.
    The matrix will nearly be transformed into a upper triangular matrix (columns left of pivot won't be reduced).
    The elements must support abs (numerical return type such that numpy can compare them), division, multiplication, subtraction, negation & have a zero constructor.
    `criterion` will be given the pivot elements and it must return `True` if a division by the element is stable.
    If no non-zero pivot element is found type(A[i,i])(0) is returned.
    """
    #https://en.wikipedia.org/wiki/Gaussian_elimination#Computing_determinants
    s = True
    for i in range(A.shape[0]):
        #pivot
        i_max, j_max = \
                np.unravel_index(np.argmax(abs(A[i:, i:])), A[i:, i:].shape)
        if not criterion(A[i+i_max, i+j_max]):
            return type(A[i+i_max, i+j_max])(0)
        swap_pivot(A, i, i+i_max, i+j_max)
        s ^= bool(i_max) != bool(j_max)
        #reduce (not left of pivot, these elements will not influence result)
        A[i+1:, i:] -= A[i, i:] * (A[i+1:, i] / A[i, i])[:, np.newaxis]
    
    d = reduce(mul, np.diag(A)) if A.dtype==np.object_ else prod(np.diag(A))
    return d if s else -d


if __name__ == '__main__':
    from fractions import Fraction
    
    def randf(shape=None, precision=1000):
        if shape == None:
            n = np.random.randint(-precision, +precision+1)
            d = np.random.randint(1, precision)
            return Fraction(n, d)
        else:
            a = np.empty(shape, dtype=np.object_)
            for i in np.ndindex(*a.shape):
                a[*i] = randf(precision=precision)
            return a
    
    for _ in range(100):
        N = np.random.randint(1, 10)
        A = randf(shape=(N, N))
        
        actual = np.linalg.det(A.astype(np.float64))
        prediction = det_gauss(A)
        assert np.isclose(float(prediction), actual)



def inv_gauss(A, criterion=bool):
    """Return the inverse of A.
    
    Calculates the inverse of A by Gaussian elimination with complete pivoting.
    The matrix will be transformed into the identity matrix.
    The elements must support abs (numerical return type such that numpy can compare them), division, multiplication & subtraction.
    `criterion` will be given the pivot elements and it must return `True` if a division by the element is stable.
    """
    #https://math.stackexchange.com/a/744213/1170417
    P, Q = np.eye(A.shape[0], dtype=A.dtype), np.eye(A.shape[0], dtype=A.dtype)
    for i in range(A.shape[0]):
        #pivot
        i_max, j_max = \
                np.unravel_index(np.argmax(abs(A[i:, i:])), A[i:, i:].shape)
        if not criterion(A[i+i_max, i+j_max]):
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
        N = np.random.randint(1, 10)
        A = randf(shape=(N, N))
        
        actual = np.linalg.inv(A.astype(np.float64))
        prediction = inv_gauss(A)
        assert np.allclose(prediction.astype(np.float64), actual)



from operator import eq
from itertools import pairwise

def is_row_echelon(A, reduced=True, comparison=eq):
    #https://stackoverflow.com/a/19502403
    pivots = [next((i for i, x in enumerate(row) if not comparison(x, 0)), A.shape[1]) for row in A]
    #check all pivots go to the right
    if not all(pi<pj or pi==A.shape[1] for pi, pj in pairwise(pivots)):
        return False
    
    if reduced:
        #check if pivots one and zeros above
        for i, p in enumerate(pivots):
            if p<A.shape[1]:
                if not comparison(A[i, p], 1) or any(not comparison(A[k, p], 0) for k in range(0, i)):
                    return False
    return True

def row_echelon(A, reduced=True, criterion=bool):
    """Transform A into (reduced) row echelon form and return the pivot column indices.
    
    Transform A into (reduced) row echelon form by Gaussian elimination with pivoting.
    The transformation happens in-place.
    The elements must support abs (numerical return type such that numpy can compare them), division, multiplication & subtraction.
    `criterion` will be given the pivot elements and it must return `True` if a division by the element is stable.
    """
    #https://en.wikipedia.org/wiki/Gaussian_elimination#Pseudocode
    i, j = 0, 0
    pivots = []
    while i<A.shape[0] and j<A.shape[1]:
        #find pivot
        if not criterion(A[(i_max := np.argmax(abs(A[i:, j])) + i), j]):
            j += 1
        else:
            #pivot
            swap_rows(A, i, i_max)
            if reduced:
                #normalize pivot
                A[i, :] /= A[i, j]
                #zeros above and below
                A[:i, :] -= A[i, :] * A[:i, j][:, np.newaxis]
                A[i+1:, :] -= A[i, :] * A[i+1:, j][:, np.newaxis]
            else:
                #zeros below
                A[i+1:, :] -= A[i, :] * (A[i+1:, j] / A[i, j])[:, np.newaxis]
            pivots += [j]
            i += 1
            j += 1
    return pivots


if __name__ == '__main__':
    for _ in range(100):
        N = np.random.randint(1, 10)
        A = randf(shape=(N, N))
        row_echelon(A, reduced=False)
        assert is_row_echelon(A, reduced=False)
        
        A = randf(shape=(N, N))
        row_echelon(A, reduced=True)
        assert is_row_echelon(A, reduced=True)
