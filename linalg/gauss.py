import numpy as np
from .util import assert_sqmatrix, posneg, _prod
from .util import swap_rows, swap_columns, swap_pivot



def det_gauss(A):
    """Return the determinant of `A`.
    
    Calculates the determinant by Gaussian elimination with complete pivoting.
    The matrix will be transformed in place into an upper triangular matrix
    (columns left of pivot won't be reduced).
    For a NxN matrix there will be
    - $N(N^2-1)/3$ subtractions (`-`),
    - $N(N^2+2)/3-1$ multiplications (`*`),
    - $N(N-1)/2$ divisions (`/`),
    - so $N(4N^2+3N-1)/6-1$ arithmetic operations in total.
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
    
    return posneg(_prod(np.diag(A)), s)

def inv_gauss(A):
    """Return the inverse of `A`.
    
    Calculates the inverse by Gaussian elimination with complete pivoting.
    The matrix will be transformed in place into the identity matrix.
    For a NxN matrix there will be
    - $N^3-N^2$ additions (`+`),
    - $2N^3-3N^2+2N-1$ subtractions (`-`),
    - $2N^3-2N^2$ multiplications (`*`),
    - $2N^2-N+1$ divisions (`/`),
    - so $5N^3-4N^2+N$ arithmetic operations in total.
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
