from .util import assert_sqmatrix, posneg, _sum, _prod



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
    parity = True
    yield tuple(pool[i] for i in indices[:r]), parity
    
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                if ((n-i) & 1) == 0:
                    parity = not parity
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                parity = not parity
                yield tuple(pool[i] for i in indices[:r]), parity
                break
        else:
            return

def det_leibniz(A):
    """Return the determinant of `A`.
    
    Calculates the determinant by the Leibniz formula.
    For a NxN matrix there will be
    - $N!-1$ additions (`+`),
    - $(N-1)N!$ multiplications (`*`),
    - so $NN!-1$ arithmetic operations in total.
    """
    assert_sqmatrix(A)
    return _sum(posneg(_prod(A[tuple(range(len(s))), s]), p) \
            for s, p in _permutations(range(A.shape[0])))
