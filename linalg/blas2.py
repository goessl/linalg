"""Vectorial operations continued.

Higher level functions that can not be directly implemented by
[`ufunc_with_cb`][linalg.blas.ufunc_with_cb].
"""



from .progress import visualisable, Progress
import numpy as np
from numpy.typing import ArrayLike, NDArray
from iteration import MISSING
from typing import Any, Never



__all__ = (
    'dot_sanitise', 'dot_announce', 'dot',
    'outer_sanitise', 'outer_announce', 'outer',
    'matmul_sanitise', 'matmul_announce', 'matmul'
)



def dot_sanitise(a: ArrayLike, b: ArrayLike, *, zero: Any=MISSING) \
        -> tuple[tuple[NDArray,NDArray],dict[str,Any]]:
    """`dot` sanitiser.
    
    See also
    --------
    - [`dot`][linalg.blas2.dot]
    """
    a, b = np.asarray(a), np.asarray(b)
    if not (a.ndim==b.ndim==1 and a.size==b.size):
        raise ValueError('a & b must be one dimensional and of same length')
    
    if zero is MISSING:
        #don't use .item(), would unpack the numpy type to a Python type
        zero = np.zeros((), dtype=np.result_type(a, b))[()]
    
    return [a, b], {'zero':zero}

def dot_announce(a: NDArray, b: NDArray, *, zero: Any=0) \
        -> dict[str,int]:
    """`dot` announcer.
    
    See also
    --------
    - [`dot`][linalg.blas2.dot]
    """
    return {
        'add': max(a.size-1, 0),
        'mul': a.size
    }

@visualisable(dot_announce, dot_sanitise)
def dot(a: NDArray, b: NDArray, *, zero: Any, progress: Progress) -> Any:
    r"""Return the dot product of two vectors.
    
    $$
        \vec{v}\cdot\vec{w} \qquad \mathbb{K}^N\times\mathbb{K}^N\to\mathbb{K} \quad N\geq0
    $$
    
    Parameters
    ----------
    a, b : numpy.typing.ArrayLike
        One dimensional arrays of same length.
    zero : Any = iteration.MISSING
        Zero element.
    progress : Iterable[str]|bool = False
        Progress visualisation specification.
    
    Returns
    -------
    Any
        Dot product.
    
    Edge cases
    ----------
    For an empty dot product ($N=0$) on object arrays provide a `zero` argument
    to not get an type unspecific `int(0)` back.
    
    Complexity
    ----------
    There will be
    
    - $\max\{N-1,0\}$ scalar additions (`add`) &
    - $N$ scalar multiplications (`mul`).
    
    See also
    --------
    - [`dot_sanitise`][linalg.blas2.dot_sanitise]
    - [`dot_announce`][linalg.blas2.dot_announce]
    """
    return progress.sumprod_default(a, b, default=zero)


def outer_sanitise(a: ArrayLike, b: ArrayLike) \
        -> tuple[tuple[NDArray,NDArray],dict[Never,Never]]:
    """`outer` sanitiser.
    
    See also
    --------
    - [`outer`][linalg.blas2.outer]
    """
    a, b = np.asarray(a), np.asarray(b)
    if not a.ndim == b.ndim == 1:
        raise ValueError('a & b must be one dimensional')
    return [a, b], {}

def outer_announce(a: NDArray, b: NDArray) -> dict[str,int]:
    """`outer` announcer.
    
    See also
    --------
    - [`outer`][linalg.blas2.outer]
    """
    return {'mul': a.size * b.size}

@visualisable(outer_announce, outer_sanitise)
def outer(a: NDArray, b: NDArray, *, progress: Progress) -> NDArray:
    r"""Return the outer product of two vectors without conjugation.
    
    $$
        \vec{v}\vec{w}^T \qquad \mathbb{K}^M\times\mathbb{K}^N\to\mathbb{K}^{M\times N} \quad M,N\geq0
    $$
    
    Parameters
    ----------
    a, b : numpy.typing.ArrayLike
        One dimensional arrays of same length.
    progress : Iterable[str]|bool = False
        Progress visualisation specification.
    
    Returns
    -------
    numpy.typing.NDArray
        Outer product.
    
    Complexity
    ----------
    There will be
    
    - $MN$ scalar multiplications (`mul`).
    
    See also
    --------
    - [`outer_sanitise`][linalg.blas2.outer_sanitise]
    - [`outer_announce`][linalg.blas2.outer_announce]
    """
    r = np.empty((a.size, b.size), np.result_type(a, b))
    for i, j in np.ndindex(r.shape):
        r[i, j] = progress.mul(a[i], b[j])
    return r


def matmul_sanitise(A: ArrayLike, B: ArrayLike, *, zero: Any=0) \
        -> tuple[tuple[NDArray,NDArray],dict[str,Any]]:
    """`matmul` sanitiser.
    
    See also
    --------
    - [`matmul`][linalg.blas2.matmul]
    """
    A, B = np.asarray(A), np.asarray(B)
    if not A.ndim == B.ndim == 2:
        raise ValueError('A & B must be two dimensional')
    if not A.shape[1] == B.shape[0]:
        raise ValueError('width of A must be height of B')
    return [A, B], {'zero':zero}

def matmul_announce(A: NDArray, B: NDArray, *, zero: Any=0) \
        -> dict[str,int]:
    """`matmul` announcer.
    
    See also
    --------
    - [`matmul`][linalg.blas2.matmul]
    """
    L, M, N = A.shape[0], A.shape[1], B.shape[1]
    return {
        'add': L * max(M-1, 0) * N,
        'mul': L * M * N
    }

@visualisable(matmul_announce, matmul_sanitise)
def matmul(A: NDArray, B: NDArray, *, zero: Any, progress: Progress) \
        -> NDArray:
    r"""Return the product of two matrices.
    
    $$
        AB \qquad \mathbb{K}^{L \times M}\times\mathbb{K}^{M \times N}\to\mathbb{K}^{L\times N} \quad L,M,N\geq0
    $$
    
    Parameters
    ----------
    A, B : numpy.typing.ArrayLike
        Two matrices with conforming shapes.
    zero : Any = 0
        Zero element.
    progress : Iterable[str]|bool = False
        Progress visualisation specification.
    
    Returns
    -------
    numpy.typing.NDArray
        Matrix product.
    
    Edge cases
    ----------
    For an empty matrix product ($M=0$) on object arrays provide a `zero`
    argument to not get type unspecific `int(0)` elements.
    
    Complexity
    ----------
    There will be
    
    - $L\max\{M-1,0\}N$ scalar additions (`add`) &
    - $LMN$ scalar multiplications (`mul`).
    
    See also
    --------
    - [`matmul_sanitise`][linalg.blas2.matmul_sanitise]
    - [`matmul_announce`][linalg.blas2.matmul_announce]
    """
    r = np.empty((A.shape[0], B.shape[1]), np.result_type(A, B))
    for i, j in np.ndindex(r.shape):
        r[i, j] = progress.sumprod_default(A[i,:], B[:,j], default=zero)
    return r
