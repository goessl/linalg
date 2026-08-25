"""Leibniz formula for determinants."""



from math import factorial
import numpy as np
from numpy.typing import ArrayLike, NDArray
from iteration import MISSING
from .progress import Progress, visualisable
from typing import Any
from collections.abc import Iterable, Generator



__all__ = (
    'permutations',
    'det_leibniz_sanitise', 'det_leibniz_announce', 'det_leibniz_cost',
    'det_leibniz'
)



def permutations[T](iterable: Iterable[T], r: int|None=None) \
        -> Generator[tuple[tuple[T,...],bool]]:
    """`itertools.permutation` including parity.
    
    Parameters
    ----------
    iterable : Iterable[T]
        Elements.
    r : int|None
        Permutation length.
    
    Yields
    ------
    tuple[tuple[T,...], bool]
        Permutations and parity.
    
    References
    ----------
    - [Python documentation - itertools - permutations](https://docs.python.org/3/library/itertools.html#itertools.permutations)
    - [stackoverflow - Finding determinant of a nxn matrix using Leibniz formula on python](https://stackoverflow.com/a/69210050)
    """
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


def det_leibniz_sanitise(A: ArrayLike, *, one: Any=MISSING) \
        -> tuple[tuple[NDArray],dict[str,Any]]:
    """`det_leibniz` sanitiser.
    
    See also
    --------
    - [`det_leibniz`][linalg.leibniz.det_leibniz]
    """
    A = np.asarray(A)
    if not (A.ndim==2 and A.shape[0]==A.shape[1]):
        raise ValueError('A must be two dimensional and square')
    
    if one is MISSING:
        #don't use .item(), would unpack the numpy type to a Python type
        one = np.ones((), dtype=A.dtype)[()]
    
    return (A,), {'one':one}

def det_leibniz_cost(N: int) -> dict[str,int]:
    """`det_leibniz` operation cost calculation.
    
    See also
    --------
    - [`det_leibniz`][linalg.leibniz.det_leibniz]
    """
    return {
        'pos': (factorial(N)+1) // 2,
        'neg': factorial(N) // 2,
        'add': factorial(N) - 1,
        'mul': max(N-1, 0) * factorial(N)
    }

def det_leibniz_announce(A: NDArray, *, one: Any=1) -> dict[str,int]:
    """`det_leibniz` announcer.
    
    See also
    --------
    - [`det_leibniz`][linalg.leibniz.det_leibniz]
    """
    return det_leibniz_cost(A.shape[0])

@visualisable(det_leibniz_announce, det_leibniz_sanitise)
def det_leibniz[T,U](A: NDArray[T], *, one: U, progress: Progress) -> T|U:
    r"""Return the determinant.
    
    $$
        \det A \qquad \mathbb{K}^{N\times N}\to\mathbb{K} \quad N\geq0
    $$
    
    Uses the Leibniz formula.
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Square matrix.
    one : Any = MISSING
        One element.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    Any
        Determinant.
    
    Edge cases
    ----------
    For a $0 \times 0$ object matrix provide a `one` argument
    to not get a type unspecific `int(1)` back.
    
    Complexity
    ----------
    There will be
    
    - $\lceil\frac{N!}{2}\rceil$ scalar affirmations (`pos`),
    - $\lfloor\frac{N!}{2}\rfloor$ scalar negations (`neg`),
    - $N!-1$ scalar additions (`add`) &
    - $(N-1)n!$ scalar multiplications (`mul`).
    
    TODO: Filter zero products.
    
    See also
    --------
    - [`det_leibniz_sanitise`][linalg.leibniz.det_leibniz_sanitise]
    - [`det_leibniz_announce`][linalg.leibniz.det_leibniz_announce]
    - [`det_leibniz_cost`][linalg.leibniz.det_leibniz_cost]
    
    References
    ----------
    [Wikipedia - Leibniz formula for determinants](https://en.wikipedia.org/wiki/Leibniz_formula_for_determinants)
    """
    i = tuple(range(A.shape[0]))
    return progress.sum_default(
            progress.posneg(progress.prod_default(A[i, p], default=one), s)
            for p, s in permutations(range(A.shape[0]))
    )
