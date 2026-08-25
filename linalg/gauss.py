"""Gaussian elimination."""



from itertools import pairwise
import numpy as np
from numpy.typing import ArrayLike, NDArray
from iteration import MISSING
from .blas import vsub, vtruediv
from .blas2 import outer
from .util import swap_rows, swap_pivot
from .progress import Progress, visualisable
from typing import Any, Never



__all__ = (
    'det_gauss_sanitise', 'det_gauss_announce', 'det_gauss_cost', 'det_gauss',
    'inv_gauss_sanitise', 'inv_gauss_announce', 'inv_gauss_cost', 'inv_gauss',
    'is_ref',
    'ref_gauss_sanitise', 'ref_gauss_announce', 'ref_gauss_cost', 'ref_gauss'
)



def det_gauss_sanitise(A: ArrayLike, *, one: Any=MISSING) \
        -> tuple[tuple[NDArray],dict[str,Any]]:
    """`det_gauss` sanitiser.
    
    See also
    --------
    - [`det_gauss`][linalg.gauss.det_gauss]
    """
    A = np.asarray(A)
    if not (A.ndim==2 and A.shape[0]==A.shape[1]):
        raise ValueError('A must be two dimensional and square')
    
    if one is MISSING:
        #don't use .item(), would unpack the numpy type to a Python type
        one = np.ones((), dtype=A.dtype)[()]
    
    return (A,), {'one':one}

def det_gauss_cost(N: int) -> dict[str,int]:
    """`det_gauss` operation cost calculation.
    
    See also
    --------
    - [`det_gauss`][linalg.gauss.det_gauss]
    """
    return {
        'pos': 1,
        'neg': 1,
        'sub': N*(N**2-1) // 3,
        'mul': max(N*(N**2+2)//3-1, 0),
        'truediv': N*(N-1) // 2
    }

def det_gauss_announce(A: NDArray, *, one: Any=1) -> dict[str,int]:
    """`det_gauss` announcer.
    
    See also
    --------
    - [`det_gauss`][linalg.gauss.det_gauss]
    """
    return det_gauss_cost(A.shape[0])

@visualisable(det_gauss_announce, det_gauss_sanitise)
def det_gauss[T,U](A: NDArray[T], *, one: U, progress: Progress) -> T|U:
    r"""Return the determinant.
    
    $$
        \det A \qquad \mathbb{K}^{N\times N}\to\mathbb{K} \quad N\geq0
    $$
    
    Uses Gaussian elimination with complete pivoting.
    
    Don't use with data types that don't divide exactly.
    
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
    
    Notes
    -----
    The matrix will be wrapped with [`numpy.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)
    and transformed in-place into an upper triangular matrix
    (columns left of pivot won't be reduced).
    
    Complexity
    ----------
    There will be
    
    - $1$ scalar affirmation (`pos`) or negation (`neg`),
    - $\frac{N(N^2-1)}{3}$ scalar subtractions (`sub`),
    - $\begin{cases} \frac{N(N^2+2)}{3}-1 & N>0 \\ 0 & N=0 \end{cases}$
    scalar multiplications (`mul`) &
    - $\frac{N(N-1)}{2}$ scalar true divisions (`truediv`).
    
    See also
    --------
    - [`det_gauss_sanitise`][linalg.gauss.det_gauss_sanitise]
    - [`det_gauss_announce`][linalg.gauss.det_gauss_announce]
    - [`inv_gauss_cost`][linalg.gauss.inv_gauss_cost]
    
    References
    ----------
    [Wikipedia - Gaussian elimination - Computing determinants](https://en.wikipedia.org/wiki/Gaussian_elimination#Computing_determinants)
    """
    N = A.shape[0]
    s = True
    for i in range(N):
        #pivot
        i_max, j_max = \
                np.unravel_index(np.argmax(np.abs(A[i:, i:])), A[i:, i:].shape)
        if not A[i+i_max, i+j_max]: #determinant zero, early exit
            M = N - i
            progress.update('pos', 1)
            progress.update('neg', 1)
            progress.update('sub', M*(M**2-1)//3)
            progress.update('mul', M*(M**2-1)//3+max(N-1, 0))
            progress.update('truediv', M*(M-1)//2)
            return A[i+i_max, i+j_max]
        swap_pivot(A, i, i+i_max, i+j_max)
        s ^= bool(i_max) != bool(j_max)
        #reduce (not left of pivot, these elements will not influence result)
        #A[i+1:, i:] -= A[i, i:] * (A[i+1:, i] / A[i, i])[:, np.newaxis]
        A[i+1:, i:] = vsub(A[i+1:, i:],
            outer(vtruediv(A[i+1:, i], A[i, i], progress=progress),
                A[i, i:], progress=progress), progress=progress)
    progress.update('neg' if s else 'pos')
    return progress.posneg(progress.prod_default(np.diag(A), default=one), s)


def inv_gauss_sanitise(A: ArrayLike) \
        -> tuple[tuple[NDArray],dict[Never,Never]]:
    """`inv_gauss` sanitiser.
    
    See also
    --------
    - [`inv_gauss`][linalg.gauss.inv_gauss]
    """
    A = np.asarray(A)
    if not (A.ndim==2 and A.shape[0]==A.shape[1]):
        raise ValueError('A must be two dimensional and square')
    return (A,), {}

def inv_gauss_cost(N: int) -> dict[str,int]:
    """`inv_gauss` operation cost calculation.
    
    See also
    --------
    - [`inv_gauss`][linalg.gauss.inv_gauss]
    """
    return {
        'sub': N**2 * (2*N-2),
        'mul': N**2 * (2*N-2),
        'truediv': 2 * N**2
    }

def inv_gauss_announce(A: NDArray) -> dict[str,int]:
    """`inv_gauss` announcer.
    
    See also
    --------
    - [`inv_gauss`][linalg.gauss.inv_gauss]
    """
    return inv_gauss_cost(A.shape[0])

@visualisable(inv_gauss_announce, inv_gauss_sanitise)
def inv_gauss[T](A: NDArray[T], *, progress: Progress) -> NDArray[T]:
    r"""Return the inverse.
    
    $$
        A^{-1} \qquad \mathbb{K}^{N\times N}\to\mathbb{K}^{N\times N} \quad N\geq0
    $$
    
    Uses Gaussian elimination with complete pivoting.
    
    Don't use with data types that don't divide exactly.
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Square matrix.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    numpy.typing.NDArray
        Inverse matrix.
    
    Raises
    ------
    ZeroDivisionError
        If `A` is singular.
    
    Notes
    -----
    The matrix will be wrapped with [`numpy.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)
    and transformed in-place into the identity matrix.
    
    Complexity
    ----------
    There will be
    
    - $2N^3-2N^2$ scalar subtractions (`sub`),
    - $2N^3-2N^2$ scalar multiplications (`mul`) &
    - $2N^2$ scalar true divisions (`truediv`).
    
    TODO: Still unnecessary operations like `A[i, i]/A[i, i]`.
    
    See also
    --------
    - [`inv_gauss_sanitise`][linalg.gauss.inv_gauss_sanitise]
    - [`inv_gauss_announce`][linalg.gauss.inv_gauss_announce]
    - [`inv_gauss_cost`][linalg.gauss.inv_gauss_cost]
    
    References
    ----------
    [StackExchange - Gauss-Jordan: Effect of column pivoting on result matrix](https://math.stackexchange.com/a/744213/1170417)
    """
    N = A.shape[0]
    P = np.eye(A.shape[0], dtype=A.dtype)
    Q = list(range(A.shape[0]))
    
    for i in range(N):
        #pivot
        i_max, j_max = \
                np.unravel_index(np.argmax(np.abs(A[i:, i:])), A[i:, i:].shape)
        if not A[i+i_max, i+j_max]: #early exit
            M = N - i
            progress.update('sub', 2*M*N*(N-1))
            progress.update('mul', 2*M*N*(N-1))
            progress.update('truediv', 2*M*N)
            raise ZeroDivisionError('matrix is singular')
        swap_pivot(A, i, i+i_max, i+j_max)
        swap_rows(P, i, i+i_max)
        #swap_columns(Q, i, i+j_max)
        Q[i], Q[i+j_max] = Q[i+j_max], Q[i]
        
        #normalize pivot
        #P[i, :] /= A[i, i]
        #A[i, :] /= A[i, i]
        P[i, :] = vtruediv(P[i, :], A[i, i], progress=progress)
        A[i, :] = vtruediv(A[i, :], A[i, i], progress=progress)
        
        #zeros above and below
        #P[:i, :] -= P[i, :] * A[:i, i][:, np.newaxis]
        #A[:i, :] -= A[i, :] * A[:i, i][:, np.newaxis]
        #P[i+1:, :] -= P[i, :] * A[i+1:, i][:, np.newaxis]
        #A[i+1:, :] -= A[i, :] * A[i+1:, i][:, np.newaxis]
        P[:i, :] = vsub(P[:i, :],
                        outer(A[:i, i], P[i, :], progress=progress),
                        progress=progress)
        A[:i, :] = vsub(A[:i, :],
                        outer(A[:i, i], A[i, :], progress=progress),
                        progress=progress)
        P[i+1:, :] = vsub(P[i+1:, :],
                          outer(A[i+1:, i], P[i, :], progress=progress),
                          progress=progress)
        A[i+1:, :] = vsub(A[i+1:, :],
                          outer(A[i+1:, i], A[i, :], progress=progress),
                          progress=progress)
    
    #return matmul(Q, P, progress)
    return P[np.argsort(Q),:]


def is_ref(A: ArrayLike, reduced: bool=True) -> bool:
    """Return if `A` is of (reduced) row echelon form.
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Matrix.
    reduced : bool = True
        Whether the reduced row echelon form should be tested.
    
    Returns
    -------
    bool
        Whether `A` is (reduced) row echelon form
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be two dimensional')
    
    pivots = [next((i for i, a in enumerate(r) if a), A.shape[1]) for r in A]
    #check all pivots ascend to the right
    if not all(pi<pj or pj==A.shape[1] for pi, pj in pairwise(pivots)):
        return False
    
    if reduced:
        #check if pivots one and zeros above
        for i, p in enumerate(pivots):
            if p<A.shape[1]:
                if A[i, p]!=1 or np.any(A[:i, p]):
                    return False
    return True


def ref_gauss_sanitise(A: NDArray, reduced: bool=True) \
        -> tuple[tuple[NDArray,bool],dict[Never,Never]]:
    """`ref_gauss` sanitiser.
    
    See also
    --------
    - [`ref_gauss`][linalg.gauss.ref_gauss]
    """
    if not isinstance(A, np.ndarray):
        raise TypeError('A must be a numpy.ndarray')
    if A.ndim != 2:
        raise ValueError('A must be two dimensional')
    return (A, reduced), {}

def ref_gauss_cost(M: int, N: int, R: int|None=None, reduced: bool=True) \
        -> dict[str,int]:
    """`ref_gauss` operation cost calculation.
    
    See also
    --------
    - [`ref_gauss`][linalg.gauss.ref_gauss]
    """
    if R is None:
        R = min(M, N)
    return {
        'sub': N*R*(M-1),
        'mul': N*R*(M-1),
        'truediv': N*R
    } if reduced else {
        'sub': N*R*(2*M-R-1) // 2,
        'mul': N*R*(2*M-R-1) // 2,
        'truediv': R*(2*M-R-1) // 2
    }

def ref_gauss_announce(A: NDArray, reduced: bool=True) -> dict[str,int]:
    """`ref_gauss` announcer.
    
    See also
    --------
    - [`ref_gauss`][linalg.gauss.ref_gauss]
    """
    return ref_gauss_cost(*A.shape, reduced=reduced)

@visualisable(ref_gauss_announce, ref_gauss_sanitise)
def ref_gauss(A: NDArray, reduced: bool=True, *, progress: Progress) \
        -> list[int]:
    r"""Transform to (reduced) row echelon form.
    
    $$
        \mathbb{K}^{M\times N}\mapsto\mathbb{K}^{M\times N} \qquad \operatorname{rank}A=R
    $$
    
    Uses Gaussian elimination with pivoting.
    
    Transforms `A` in-place (therefore always pass a `numpy.typing.NDArray`).
    
    Parameters
    ----------
    A : numpy.typing.NDArray
        Matrix.
    reduced : bool = True
        Whether should be transformed into reduced row echelon form.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    list[int]
        Pivot indices.
    
    Complexity
    ----------
    There will be (unreduced)
    
    - $NR(2M-R-1)/2$ scalar subtractions (`sub`),
    - $NR(2M-R-1)/2$ scalar multiplications (`mul`) &
    - $R(2M-R-1)/2$ scalar true divisions (`truediv`).
    
    or (reduced)
    
    - $NR(M-1)$ scalar subtractions (`sub`),
    - $NR(M-1)$ scalar multiplications (`mul`) &
    - $NR$ scalar true divisions (`truediv`).
    
    See also
    --------
    - [`ref_gauss_sanitise`][linalg.gauss.ref_gauss_sanitise]
    - [`ref_gauss_announce`][linalg.gauss.ref_gauss_announce]
    
    References
    ----------
    [Wikipedia - Gaussian elimination - Pseudocode](https://en.wikipedia.org/wiki/Gaussian_elimination#Pseudocode)
    """
    M, N = A.shape
    i, j = 0, 0
    R, lost = min(M, N), 0
    pivots = []
    while i<M and j<N:
        #find pivot
        if not A[(p := np.argmax(np.abs(A[i:, j])) + i), j]:
            j += 1
            if N-j < M-i:
                lost += 1
                if reduced:
                    progress.update('sub', N*(M-1))
                    progress.update('mul', N*(M-1))
                    progress.update('truediv', N)
                else:
                    k = R - lost
                    progress.update('sub', N*(M-k-1))
                    progress.update('mul', N*(M-k-1))
                    progress.update('truediv', M-k-1)
        else:
            #pivot
            swap_rows(A, i, p)
            if reduced:
                #normalize pivot
                #A[i, :] /= A[i, j]
                A[i, :] = vtruediv(A[i, :], A[i, j], progress=progress)
                #zeros above and below
                #A[:i, :] -= A[i, :] * A[:i, j][:, np.newaxis]
                #A[i+1:, :] -= A[i, :] * A[i+1:, j][:, np.newaxis]
                A[:i, :] = vsub(A[:i, :],
                                outer(A[:i, j], A[i, :], progress=progress),
                                progress=progress)
                A[i+1:, :] = vsub(A[i+1:, :],
                                  outer(A[i+1:, j], A[i, :], progress=progress),
                                  progress=progress)
            else:
                #zeros below
                #A[i+1:, :] -= A[i, :] * (A[i+1:, j] / A[i, j])[:, np.newaxis]
                A[i+1:, :] = vsub(A[i+1:, :],
                                  outer(vtruediv(A[i+1:, j],
                                                 A[i, j],
                                                 progress=progress),
                                        A[i, :],
                                        progress=progress),
                                  progress=progress)
            pivots.append(j)
            i += 1
            j += 1
    return pivots
