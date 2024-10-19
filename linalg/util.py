import numpy as np
from functools import reduce
from operator import add, mul



def posneg(x, s):
    """Return `+x` if `s==True` or `-x` if `s==False`.
    
    To use the unary operators `+` (`__pos__`) & `-` (`__neg__`)
    instead of multiplication with `+1` & `-1` (`__mul__`)
    as the univariate operators may be more optimised
    and multiplication with integers doesn't have to be implemented
    (might be the case for custom prototyped number types).
    """
    return +x if s else -x


def _sum(iterable):
    """Like `sum` but without initial '+0'.
    
    Avoids initial '+int(0)'
    for correct operation counting in complexity analysis.
    For `float`s keep using `sum` for better precision.
    See [README.md#todo](README.md#Conventions) for justification.
    """
    return reduce(add, iterable)

def _prod(iterable):
    """Like `math.prod` but for non-numeric types.
    
    `math.prod` might reject non-numeric types:
    https://docs.python.org/3/library/math.html#math.prod.
    For `float`s keep using `math.prod` for better precision.
    See [README.md#todo](README.md#Conventions) for justification.
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
