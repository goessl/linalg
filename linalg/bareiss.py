import numpy as np
from .util import assert_sqmatrix, posneg, swap_pivot



def det_bareiss(A):
    """Return the determinant of an integer matrix `A`.
    
    Calculates the determinant by the Bareiss algorithm.
    Transforms `A` in place.
    For a NxN-matrix there will be
    - $N(N^2-1)/3$ subtractions (`-`),
    - $2N(N^2-1)/3$ multiplications (`*`),
    - $N(N^2-3N+2)/3$ floor divisions (`//`),
    - so $N(4N^2-3N-1)/3$ operations in total.
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
        #A[i+1:, i:]*A[i, i] instead of A[i, i]*A[i+1:, i:]
        #to keep CounterWrapper.__mul__ from wrapping an array
        A[i+1:, i:] = \
                (A[i+1:, i:]*A[i, i] - A[i+1:, i][:, np.newaxis]*A[i, i:]) \
                // (A[i-1, i-1] if i>0 else 1)
    
    return posneg(A[-1, -1], s)
