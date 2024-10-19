import numpy as np
from .util import assert_sqmatrix



def backsub(T, y, tril=False):
    """Return the solution for `Tx=y` where the matrix `T` is triangular.
    
    Back substitution like BLAS `xTRSV`.
    `tril=False` if `T` is upper triangular,
    `tril=True` if `T` is lower triangular.
    For a N-vector/NxN-matrix there will be
    - $N(N-3)/2+1$ additions (`+`),
    - $N$ subtractions (`-`),
    - $N(N-1)/2$ multiplications (`*`),
    - $N$ divisions (`/`),
    - so $N2+1$ operations in total.
    """
    assert_sqmatrix(T)
    x = np.empty_like(y)
    if tril:
        for i in range(len(x)):
            x[i] = (y[i] - T[i, :i]@x[:i]) / T[i, i]
    else:
        for i in reversed(range(len(x))):
            x[i] = (y[i] - T[i, i+1:]@x[i+1:]) / T[i, i]
    return x
