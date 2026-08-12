"""Vectorial operations continued.

Higher level functions that can not be directly implemented by
[`ufunc_with_cb`][linalg.blas.ufunc_with_cb].
"""



from .progress import visualisable, Progress
import numpy as np
import numpy.typing as npt
from typing import Any



__all__ = (
    'dot_ops', 'dot',
    'outer_ops', 'outer',
    'matmul_ops', 'matmul'
)



def dot_ops(a: npt.ArrayLike, b: npt.ArrayLike) -> dict[str,int]:
    size = np.size(a)
    return {'add': max(size-1, 0), 'mul':size}

@visualisable(dot_ops)
def dot(a: npt.ArrayLike, b: npt.ArrayLike, *, progress:Progress) -> Any:
    r"""Return the dot product of two vectors.
    
    $$
        \vec{v}\cdot\vec{w} \qquad \mathbb{K}^n\times\mathbb{K}^n\to\mathbb{K}
    $$
    
    Complexity
    ----------
    For two vectors of length $n$ there will be
    
    - $\max\{n-1,0\}$ scalar additions (`add`) &
    - $n$ scalar multiplications (`mul`).
    """
    a, b = np.asarray(a), np.asarray(b)
    if not (a.ndim==b.ndim==1 and a.size==b.size):
        raise ValueError('a & b must be one dimensional and of same length')
    
    return progress.sumprod_default(a, b,
            default=np.zeros((), np.result_type(a, b)).item())


def outer_ops(a: npt.ArrayLike, b: npt.ArrayLike) -> dict[str,int]:
    return {'mul': np.size(a)*np.size(b)}

@visualisable(outer_ops)
def outer(a: npt.ArrayLike, b: npt.ArrayLike, *, progress:Progress) -> Any:
    r"""Return the outer product of two vectors without conjugation.
    
    $$
        \vec{v}\vec{w}^T \qquad \mathbb{K}^m\times\mathbb{K}^n\to\mathbb{K}^{m\times n}
    $$
    
    Complexity
    ----------
    For two vectors of length $m$ & $n$ there will be
    
    - $mn$ scalar multiplications (`mul`).
    """
    a, b = np.asarray(a), np.asarray(b)
    if not a.ndim == b.ndim == 1:
        raise ValueError('a & b must be one dimensional')
    
    r = np.empty((a.size, b.size), np.result_type(a, b))
    for i, j in np.ndindex(r.shape):
        r[i, j] = progress.mul(a[i], b[j])
    return r


def matmul_ops(a: npt.ArrayLike, b: npt.ArrayLike) -> dict[str,int]:
    a, b = np.shape(a), np.shape(b)
    try:
        return {'add': a[0]*b[1]*max(a[1]-1, 0), 'mul':a[0]*a[1]*b[1]}
    except IndexError:
        return {}

@visualisable(matmul_ops)
def matmul(a: npt.ArrayLike, b: npt.ArrayLike, *, progress:Progress) -> Any:
    r"""Return the product of two matrices.
    
    $$
        AB \qquad \mathbb{K}^{l \times m}\times\mathbb{K}^{m \times n}\to\mathbb{K}^{l\times n}
    $$
    
    Complexity
    ----------
    For two matrices of sizes $l \times m$ & $m \times n$ there will be
    
    - $l\max\{m-1,0\}n$ scalar additions (`add`) &
    - $lmn$ scalar multiplications (`mul`).
    """
    a, b = np.asarray(a), np.asarray(b)
    if not a.ndim == b.ndim == 2:
        raise ValueError('a & b must be two dimensional')
    if not a.shape[1] == b.shape[0]:
        raise ValueError('width of a must be height of b')
    
    r = np.empty((a.shape[0], b.shape[1]), np.result_type(a, b))
    for i, j in np.ndindex(r.shape):
        r[i, j] = progress.sumprod_default(a[i,:], b[:,j],
            default=np.zeros((), np.result_type(a, b)).item())
    return r
