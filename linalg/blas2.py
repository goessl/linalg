"""Vectorial operations continued.

Higher level functions that can not be directly implemented by
[`ufunc_with_cb`][linalg.blas.ufunc_with_cb].
"""



from .progress import visualisable, Progress
import numpy as np
import numpy.typing as npt
from typing import Any
from collections.abc import Sequence, Mapping



__all__ = (
    'dot_sanitise', 'dot_announce', 'dot',
    'outer_sanitise', 'outer_announce', 'outer',
    'matmul_sanitise', 'matmul_announce', 'matmul'
)



def dot_sanitise(a: npt.ArrayLike, b: npt.ArrayLike) \
        -> tuple[Sequence, Mapping]:
    """`dot` sanitiser.
    
    See also
    --------
    - [`dot`][linalg.blas2.dot]
    """
    a, b = np.asarray(a), np.asarray(b)
    if not (a.ndim==b.ndim==1 and a.size==b.size):
        raise ValueError('a & b must be one dimensional and of same length')
    return [a, b], {}

def dot_announce(a: npt.NDArray, b: npt.NDArray) -> dict[str,int]:
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
def dot(a: npt.NDArray, b: npt.NDArray, *, progress:Progress) -> Any:
    r"""Return the dot product of two vectors.
    
    $$
        \vec{v}\cdot\vec{w} \qquad \mathbb{K}^n\times\mathbb{K}^n\to\mathbb{K}
    $$
    
    Parameters
    ----------
    a, b : numpy.typing.ArrayLike
        One dimensional arrays of same length.
    progress : Iterable[str]|bool = False
        Progress visualisation specification.
    
    Returns
    -------
    Any
        Dot product.
    
    Complexity
    ----------
    For two vectors of length $n$ there will be
    
    - $\max\{n-1,0\}$ scalar additions (`add`) &
    - $n$ scalar multiplications (`mul`).
    
    See also
    --------
    - [`dot_sanitise`][linalg.blas2.dot_sanitise]
    - [`dot_announce`][linalg.blas2.dot_announce]
    """
    return progress.sumprod_default(a, b,
            default=np.zeros((), np.result_type(a, b)).item())


def outer_sanitise(a: npt.ArrayLike, b: npt.ArrayLike) \
        -> tuple[Sequence, Mapping]:
    """`outer` sanitiser.
    
    See also
    --------
    - [`outer`][linalg.blas2.outer]
    """
    a, b = np.asarray(a), np.asarray(b)
    if not a.ndim == b.ndim == 1:
        raise ValueError('a & b must be one dimensional')
    return [a, b], {}

def outer_announce(a: npt.NDArray, b: npt.NDArray) -> dict[str,int]:
    """`outer` announcer.
    
    See also
    --------
    - [`outer`][linalg.blas2.outer]
    """
    return {'mul': a.size * b.size}

@visualisable(outer_announce, outer_sanitise)
def outer(a: npt.NDArray, b: npt.NDArray, *, progress:Progress) -> npt.NDArray:
    r"""Return the outer product of two vectors without conjugation.
    
    $$
        \vec{v}\vec{w}^T \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{m\times n}
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
    For two vectors of length $m$ & $n$ there will be
    
    - $mn$ scalar multiplications (`mul`).
    
    See also
    --------
    - [`outer_sanitise`][linalg.blas2.outer_sanitise]
    - [`outer_announce`][linalg.blas2.outer_announce]
    """
    r = np.empty((a.size, b.size), np.result_type(a, b))
    for i, j in np.ndindex(r.shape):
        r[i, j] = progress.mul(a[i], b[j])
    return r


def matmul_sanitise(A: npt.ArrayLike, B: npt.ArrayLike) \
        -> tuple[Sequence, Mapping]:
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
    return [A, B], {}

def matmul_announce(A: npt.NDArray, B: npt.NDArray) -> dict[str,int]:
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
def matmul(A: npt.NDArray, B: npt.NDArray, *, progress:Progress) \
        -> npt.NDArray:
    r"""Return the product of two matrices.
    
    $$
        AB \qquad \mathbb{K}^{l \times m}\times\mathbb{K}^{m \times n}\to\mathbb{K}^{l\times n}
    $$
    
    Parameters
    ----------
    A, B : numpy.typing.ArrayLike
        Two matrices with conforming shapes.
    progress : Iterable[str]|bool = False
        Progress visualisation specification.
    
    Returns
    -------
    numpy.typing.NDArray
        Matrix product.
    
    Complexity
    ----------
    For two matrices of sizes $l \times m$ & $m \times n$ there will be
    
    - $l\max\{m-1,0\}n$ scalar additions (`add`) &
    - $lmn$ scalar multiplications (`mul`).
    
    See also
    --------
    - [`matmul_sanitise`][linalg.blas2.matmul_sanitise]
    - [`matmul_announce`][linalg.blas2.matmul_announce]
    """
    r = np.empty((A.shape[0], B.shape[1]), np.result_type(A, B))
    zero = np.zeros((), np.result_type(A, B)).item()
    for i, j in np.ndindex(r.shape):
        r[i, j] = progress.sumprod_default(A[i,:], B[:,j], default=zero)
    return r
