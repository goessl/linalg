"""Vectorial operations continued.

Higher level functions that can not be directly implemented by
[`ufunc_with_cb`][linalg.blas.ufunc_with_cb].
"""



from itertools import pairwise
from functools import cache
from .util import dict_add, dict_iadd
from .progress import visualisable, Progress
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Never
from types import MappingProxyType



__all__ = (
    'matmul_sanitise', 'matmul_announce', 'matmul_promote', 'matmul_cost',
    'matmul',
    'outer_sanitise', 'outer_announce', 'outer_cost', 'outer',
    'matmulchain_sanitise', 'matmulchain_announce', 'matmulchain_plan',
    'matmulchain'
)



def matmul_sanitise(A: ArrayLike, B: ArrayLike, *, zero: Any=0) \
        -> tuple[tuple[NDArray,NDArray],dict[str,Any]]:
    """`matmul` sanitiser.
    
    See also
    --------
    - [`matmul`][linalg.blas2.matmul]
    """
    A, B = np.asarray(A), np.asarray(B)
    if not {A.ndim, B.ndim} <= {1, 2}:
        raise ValueError('A & B must be one or two dimensional')
    if not A.shape[-1] == B.shape[0]:
        raise ValueError('width of A must be height of B')
    
    return (A, B), {'zero':zero}

def matmul_cost(L: int, M: int, N: int) -> dict[str,int]:
    """`matmul` operation cost calculation.
    
    See also
    --------
    - [`matmul`][linalg.blas2.matmul]
    """
    return {
        'add': L * max(M-1, 0) * N,
        'mul': L * M * N
    }

def matmul_announce(A: NDArray, B: NDArray, *, zero: Any=0) \
        -> dict[str,int]:
    """`matmul` announcer.
    
    See also
    --------
    - [`matmul`][linalg.blas2.matmul]
    """
    A, B = matmul_promote(A, B)
    return matmul_cost(A.shape[0], A.shape[1], B.shape[1])

def matmul_promote(A: NDArray, B: NDArray) -> tuple[NDArray, NDArray]:
    """Promote both matrix/vector factors to matrices."""
    return (A if A.ndim == 2 else A[np.newaxis, :],
            B if B.ndim == 2 else B[:, np.newaxis])

@visualisable(matmul_announce, matmul_sanitise)
def matmul(A: NDArray, B: NDArray, *, zero: Any, progress: Progress) \
        -> NDArray:
    r"""Return the matrix product.
    
    $$
        \begin{aligned}
            \vec{A}^T\vec{B} &\qquad \mathbb{K}^M\times\mathbb{K}^M\to\mathbb{K} \\
            \vec{A}^TB &\qquad \mathbb{K}^M\times\mathbb{K}^{M \times N}\to\mathbb{K}^N \\
            A\vec{B} &\qquad \mathbb{K}^{L \times M}\times\mathbb{K}^M\to\mathbb{K}^L \\
            AB &\qquad \mathbb{K}^{L \times M}\times\mathbb{K}^{M \times N}\to\mathbb{K}^{L\times N}
        \end{aligned} \quad L, M, N\geq0
    $$
    
    Parameters
    ----------
    A, B : numpy.typing.ArrayLike
        Matrices or vectors with conforming shapes.
    zero : Any = 0
        Zero element.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    numpy.typing.NDArray
        Product.
    
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
    - [`matmul_cost`][linalg.blas2.matmul_cost]
    """
    Ap, Bp = matmul_promote(A, B)
    r = np.empty((Ap.shape[0], Bp.shape[1]), np.result_type(A, B))
    for i, j in np.ndindex(r.shape):
        r[i, j] = progress.sumprod_default(Ap[i,:], Bp[:,j], default=zero)
    
    if A.ndim == B.ndim == 1: #vector^T x vector
        return r[0, 0]
    elif A.ndim == 1: #vector^T x matrix
        return r[0, :]
    elif B.ndim == 1: #matrix x vector
        return r[:, 0]
    return r #matrix x matrix


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
    return (a, b), {}

def outer_cost(M: int, N: int) -> dict[str,int]:
    """`outer` operation cost calculation.
    
    See also
    --------
    - [`outer`][linalg.blas2.outer]
    """
    return {'mul': M * N}

def outer_announce(a: NDArray, b: NDArray) -> dict[str,int]:
    """`outer` announcer.
    
    See also
    --------
    - [`outer`][linalg.blas2.outer]
    """
    return outer_cost(a.size, b.size)

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
    progress : Iterable[str]|bool|Progress = False
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
    - [`outer_cost`][linalg.blas2.outer_cost]
    """
    r = np.empty((a.size, b.size), np.result_type(a, b))
    for i, j in np.ndindex(r.shape):
        r[i, j] = progress.mul(a[i], b[j])
    return r


@cache
def matmulchain_plan(s: tuple[int,...]) \
        -> tuple[MappingProxyType[str,int], tuple[int,...]]:
    """`matmulchain` optimal multiplication order search.
    
    See also
    --------
    - [`matmulchain`][linalg.blas2.matmulchain]
    """
    if len(s) < 2:
        raise ValueError('expecting at least one factor')
    elif len(s) == 2:
        return MappingProxyType({'add':0, 'mul':0}), ()
    
    cost = {'mul': None}
    order = ()
    for i in range(1, len(s) - 1):
        s_remaining = s[:i] + s[i+1:]
        cost_remaining, indices_remaining = matmulchain_plan(s_remaining)
        cost_remaining = dict_add(cost_remaining, matmul_cost(*s[i-1:i+2]))
        
        if cost['mul'] is None or cost_remaining['mul']<cost['mul']:
            cost = cost_remaining
            order = (i-1,) + indices_remaining
        
    return MappingProxyType(cost), order

def matmulchain_sanitise(*As: ArrayLike, zero: Any=0,
        order: tuple[int,...]|None=None) \
        -> tuple[tuple[NDArray,...], dict[str,Any]]:
    """`matmulchain` sanitiser.
    
    See also
    --------
    - [`matmulchain`][linalg.blas2.matmulchain]
    """
    As = tuple(map(np.asarray, As))
    if not As:
        raise ValueError('at least one array required')
    if not all(a.ndim==2 for a in As):
        raise ValueError('matrices expected')
    if not all(a.shape[1]==b.shape[0] for a, b in pairwise(As)):
        raise ValueError('shapes not matching')
    
    if order is None: #sanitiser idempotency
        s = (As[0].shape[0],) + tuple(a.shape[1] for a in As)
        _, order = matmulchain_plan(s)
    
    return As, {'zero':zero, 'order':order}

def matmulchain_announce(*As: NDArray, zero: Any=0,
        order: tuple[int,...]) -> dict[str,int]:
    """`matmulchain` announcer.
    
    See also
    --------
    - [`matmulchain`][linalg.blas2.matmulchain]
    """
    s = [As[0].shape[0]] + [a.shape[1] for a in As]
    cost = {'add': 0, 'mul': 0}
    for i in order:
        dict_iadd(cost, matmul_cost(*s[i:i+3]))
        del s[i+1]
    return cost

@visualisable(matmulchain_announce, matmulchain_sanitise)
def matmulchain(*As: ArrayLike, zero: Any,
        order: tuple[int,...], progress: Progress) -> NDArray:
    r"""Return the matrix chain product in optimal multiplication order.
    
    $$
        A_1A_2\cdots A_n \qquad \mathbb{K}^{N_0 \times N_1}\times\mathbb{K}^{N_1 \times N_2}\times\cdots\times\mathbb{K}^{N_{n-1}\times N_n}\to\mathbb{K}^{N_0\times N_n} \quad N_i\geq0 \ \forall i\in\{0, 1, \dots, n\}
    $$
    
    Chooses the optimal order to minimise scalar multiplications (`mul`).
    
    Parameters
    ----------
    As : numpy.typing.ArrayLike
        Matrices.
    zero : Any = 0
        Zero element.
    order : tuple[int,...]|None = None
        Leave as `None`. The sanitiser searches for the optimal order
        and passes it to the announcer and executor.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    numpy.typing.NDArray
        Product.
    
    See also
    --------
    - [`matmulchain_sanitise`][linalg.blas2.matmulchain_sanitise]
    - [`matmulchain_announce`][linalg.blas2.matmulchain_announce]
    - [`matmulchain_plan`][linalg.blas2.matmulchain_plan]
    
    References
    ----------
    - [Wikipedia - Matrix chain multiplication](https://en.wikipedia.org/wiki/Matrix_chain_multiplication)
    """
    for i in order:
        As = As[:i] \
                 + (matmul(As[i], As[i+1], zero=zero, progress=progress),) \
                 + As[i+2:]
    return As[0]
