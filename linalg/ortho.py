"""Orthogonalisations and normalisations."""



from itertools import combinations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from iteration import sumprod_default
from .blas import vsub, vmul
from .blas2 import matmul
from .progress import visualisable, Progress
from typing import Never



__all__ = (
    'are_orthogonal', 'is_normalised', 'are_orthonormal',
    'gram_schmidt_sanitise', 'gram_schmidt_announce', 'gram_schmidt_cost',
    'gram_schmidt'
)



def are_orthogonal(*vs: ArrayLike) -> bool:
    """Return if all vectors are orthogonal.
    
    Parameters
    ----------
    vs : numpy.typing.ArrayLike
        Vectors.
    
    Returns
    -------
    bool
        Whether they are orthogonal.
    
    See also
    --------
    - [`are_orthonormal`][linalg.ortho.are_orthonormal]
    """
    vs = tuple(map(np.asarray, vs))
    if not all(v.ndim==1 for v in vs):
        raise ValueError('vs must be vectors')
    if len({v.size for v in vs}) > 1:
        raise ValueError('all vs must be of same dimensionality')
    return all(not bool(sumprod_default(v, w)) for v, w in combinations(vs, 2))

def is_normalised(v: ArrayLike) -> bool:
    """Return if a vector is normalised.
    
    Parameters
    ----------
    v : numpy.typing.ArrayLike
        Vector.
    
    Returns
    -------
    bool
        Whether it is normalised.
    
    See also
    --------
    - [`are_orthonormal`][linalg.ortho.are_orthonormal]
    """
    v = np.asarray(v)
    if v.ndim != 1:
        raise ValueError('v must be a vector')
    return sumprod_default(v, v) == 1

def are_orthonormal(*vs: ArrayLike) -> bool:
    """Return if all vectors are orthonormal.
    
    Parameters
    ----------
    vs : numpy.typing.ArrayLike
        Vectors.
    
    Returns
    -------
    bool
        Whether they are orthonormal.
    
    See also
    --------
    - [`are_orthogonal`][linalg.ortho.are_orthogonal]
    - [`is_normalised`][linalg.ortho.is_normalised]
    """
    return are_orthogonal(*vs) and all(is_normalised(v) for v in vs)


def gram_schmidt_sanitise(vs: list[ArrayLike]) \
        -> tuple[tuple[list[NDArray]],dict[Never,Never]]:
    """`gram_schmidt` sanitiser.
    
    See also
    --------
    - [`gram_schmidt`][linalg.ortho.gram_schmidt]
    """
    for i in range(len(vs)):
        vs[i] = np.asarray(vs[i])
    if not (all(v.ndim==1 for v in vs) and len({v.size for v in vs})<=1):
        raise ValueError('vs must be vector and of the same length')
    return (vs,), {}

def gram_schmidt_cost(M: int, N: int) -> dict[str,int]:
    """`gram_schmidt` operation cost calculation.
    
    See also
    --------
    - [`gram_schmidt`][linalg.ortho.gram_schmidt]
    """
    return {
        'add': M*(M+1)//2 * max(N-1, 0),
        'sub': M*(M-1)*N // 2,
        'mul': M**2 * N,
        'truediv': M*(M-1) // 2
    }

def gram_schmidt_announce(vs: list[NDArray]) -> dict[str,int]:
    """`gram_schmidt` announcer.
    
    See also
    --------
    - [`gram_schmidt`][linalg.ortho.gram_schmidt]
    """
    M = len(vs)
    N = next(iter({v.size for v in vs}), 0)
    return gram_schmidt_cost(M, N)

@visualisable(gram_schmidt_announce, gram_schmidt_sanitise)
def gram_schmidt(vs: list[NDArray], *, progress: Progress) -> None:
    r"""Orthogonalise.
    
    Transformation happens in-place.
    
    Don't use with data types that don't divide exactly.
    
    Parameters
    ----------
    vs : list[numpy.typing.ArrayLike]
        Vectors.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Raises
    ------
    ZeroDivisionError
        If the vectors can't be orthogonalised.
    
    Complexity
    ----------
    For $M$ vectors of length $N$ there will be
    
    - $\frac{M(M+1)}{2}\max\{N-1,0\}$ scalar additions (`add`),
    - $\frac{M(M-1)N}{2}$ scalar subtractions (`sub`),
    - $M^2N$ scalar multiplications (`mul`) &
    - $\frac{M(M-1)}{2}$ scalar true divisions (`truediv`).
    
    See also
    --------
    - [`gram_schmidt_sanitise`][linalg.ortho.gram_schmidt_sanitise]
    - [`gram_schmidt_announce`][linalg.ortho.gram_schmidt_announce]
    - [`gram_schmidt_cost`][linalg.ortho.gram_schmidt_cost]
    
    References
    ----------
    - [Wikipedia - Gram–Schmidt process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
    """
    dots = []
    for i in range(len(vs)):
        for j in range(i):
            #vs[i] -= vs[i] @ vs[j] / dots[j] * vs[j]
            vs[i][:] = vsub(
                    vs[i],
                    vmul(
                        progress.truediv(
                            matmul(vs[i], vs[j], progress=progress),
                            dots[j]),
                        vs[j],
                        progress=progress
                    ),
                    progress=progress
            )
        dots.append(matmul(vs[i], vs[i], progress=progress))
        if not dots[-1]:
            raise ZeroDivisionError('not orthogonalisable')
