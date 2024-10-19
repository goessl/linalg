import numpy as np
from .util import assert_matrix, swap_rows
from .gauss import inv_gauss
from itertools import pairwise



def is_ref(A, reduced=True):
    """Return if `A` is of (reduced) row echelon form."""
    assert_matrix(A)
    pivots = [next((i for i, a in enumerate(r) if a), A.shape[1]) for r in A]
    #check all pivots ascend to the right
    if not all(pi<pj or pj==A.shape[1] for pi, pj in pairwise(pivots)):
        return False
    
    if reduced:
        #check if pivots one and zeros above
        for i, p in enumerate(pivots):
            if p<A.shape[1]:
                if A[i, p]!=1 or np.any(A[:i, p]):
                    return False
    return True

def ref_gauss(A, reduced=True):
    """Transform `A` into (reduced) row echelon form.
    
    By Gaussian elimination with pivoting.
    Transforms `A` in place.
    For a MxN-matrix of rank r there will be
    - $MNr-Nr(r+1)/2$ subtractions (`-`),
    - $MNr-Nr(r+1)/2$ multiplications (`*`),
    - $Mr-r(r+1)/2$ divisions (`/`),
    - so $Mr(2N+1)-Nr(r+1)-r(r+1)/2$ operations in total.
    """
    #https://en.wikipedia.org/wiki/Gaussian_elimination#Pseudocode
    i, j = 0, 0
    pivots = []
    while i<A.shape[0] and j<A.shape[1]:
        #find pivot
        if not A[(p := np.argmax(abs(A[i:, j])) + i), j]:
            j += 1
        else:
            #pivot
            swap_rows(A, i, p)
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


def rank_decomposition(A):
    """Return a rank decomposition `B, C` of `A` such that `A=BC`."""
    #https://en.wikipedia.org/wiki/Rank_factorization#Rank_factorization_from_reduced_row_echelon_forms
    pivots = ref_gauss(C:=A.copy())
    #delete non pivot columns from A
    B = np.delete(A, [i for i in range(A.shape[1]) if i not in pivots], 1)
    #delete zero rows from C
    for i in reversed(range(C.shape[0])):
        if np.all(np.logical_not(C[i, :])):
            C = np.delete(C, i, 0)
    return B, C

def pinv(A):
    """Return the Moore–Penrose inverse of `A`."""
    #https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Rank_decomposition
    B, C = rank_decomposition(A)
    return C.T @ inv_gauss(C@C.T) @ inv_gauss(B.T@B) @ B.T

def lstsq(X, y):
    """Return the linear least squares solution `b` for `y=Xb`."""
    return pinv(X.T@X) @ X.T @ y
