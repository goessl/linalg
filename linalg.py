import numpy as np
from fractions import Fraction



def randf(shape=None, precision=1000):
    """Return a random `Fraction` or an array of random `Fraction`s.
    
    Their numerators `n` are `-precision <= n <= +precision`
    and their denominators `d` are `1 <= d <= +precision`.
    """
    n = np.random.randint(-precision, +precision+1, shape)
    d = np.random.randint(1, +precision+1, shape)
    return Fraction(n, d) if shape is None else np.vectorize(Fraction)(n, d)


def submatrix(A, i, j):
    """Return a copy of `A` without the `i`-th row and `j`-th column."""
    return np.delete(np.delete(A, i, 0), j, 1)





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
