import numpy as np
from .util import assert_sqmatrix, posneg, _sum
from .adjugate import cofactor



def det_laplace(A):
    """Return the determinant of `A`.
    
    Calculates the determinant by Laplace expansion.
    Uses the row/column with the most zero elements.
    For a NxN matrix there will be at most
    - $N!-1$ additions (`+`) and subtractions (`-`),
    - $floor((e-1)N!)-1$ multiplications (`*`) (https://oeis.org/A038156),
    - so $floor(eN!)-2$ arithmetic operations
      in total (https://oeis.org/A026243).
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
        return _sum(aij*cofactor(A, i, j, det_laplace) \
                for j, aij in enumerate(A[i,:]) if aij)
    else:
        j = np.argmin(nonzeros_in_columns)
        return _sum(aij*cofactor(A, i, j, det_laplace) \
                for i, aij in enumerate(A[:,j]) if aij)
