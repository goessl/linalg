import numpy as np
from .util import posneg, submatrix
from .gauss import det_gauss



def minor(A, i, j, det=det_gauss):
    """Return the `i,j`-th minor of `A`.
    
    Uses the given determinant algorithm.
    """
    return det(submatrix(A, i, j))

def cofactor(A, i, j, det=det_gauss):
    """Return the `i,j`-th cofactor of `A`.
    
    Uses the given determinant algorithm.
    """
    return posneg(minor(A, i, j, det), not (i+j)%2)

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
