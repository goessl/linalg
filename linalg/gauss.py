"""Gaussian elimination."""



import numpy as np
import numpy.typing as npt
from .blas import vsub, vtruediv
from .blas2 import outer
from .progress import Progress, visualisable
from typing import Any
from collections.abc import Sequence, Mapping



__all__ = (
    'swap_rows', 'swap_columns', 'swap_pivot',
    'det_gauss_sanitise', 'det_gauss_announce', 'det_gauss',
    'inv_gauss_sanitise', 'inv_gauss_announce', 'inv_gauss'
)



def swap_rows(A:npt.NDArray, i:int, j:int) -> None:
    """Swap the `i`-th and `j`-th row of `A` in-place.
    
    Parameters
    ----------
    A : numpy.typing.NDArray
        Two dimensional array.
    i, j : int
        Row indices.
    
    References
    ----------
    [stackoverflow - Swap two rows in a numpy array in python](https://stackoverflow.com/a/54069951)
    """
    if not isinstance(A, np.ndarray):
        raise TypeError('A must be a numpy.ndarray')
    if not A.ndim == 2:
        raise ValueError('A must be two dimensional')
    
    A[[i, j], :] = A[[j, i], :]

def swap_columns(A:npt.NDArray, i:int, j:int) -> None:
    """Swap the `i`-th and `j`-th column of `A` in-place.
    
    Parameters
    ----------
    A : numpy.typing.NDArray
        Two dimensional array.
    i, j : int
        Column indices.
    
    References
    ----------
    [stackoverflow - Swap two rows in a numpy array in python](https://stackoverflow.com/a/54069951)
    """
    if not isinstance(A, np.ndarray):
        raise TypeError('A must be a numpy.ndarray')
    if not A.ndim == 2:
        raise ValueError('A must be two dimensional')
    
    A[:, [i, j]] = A[:, [j, i]]

def swap_pivot(A:npt.NDArray, p:int, i:int, j:int) -> None:
    """Swap the `p`-&`i`-th rows and `p`-&`j`-th columns of `A` in-place.
    
    Parameters
    ----------
    A : numpy.typing.NDArray
        Two dimensional array.
    p, i, j : int
        Column and row indices.
    
    References
    ----------
    [stackoverflow - Swap two rows in a numpy array in python](https://stackoverflow.com/a/54069951)
    """
    if not isinstance(A, np.ndarray):
        raise TypeError('A must be a numpy.ndarray')
    if not A.ndim == 2:
        raise ValueError('A must be two dimensional')
    
    swap_rows(A, p, i)
    swap_columns(A, p, j)


def det_gauss_sanitise(A:npt.ArrayLike) -> tuple[Sequence, Mapping]:
    """`det_gauss` sanitiser.
    
    See also
    --------
    - [`det_gauss`][linalg.gauss.det_gauss]
    """
    A = np.asarray(A)
    if not (A.ndim==2 and A.shape[0]==A.shape[1]):
        raise ValueError('A must be two dimensional and square')
    return [A], {}

def det_gauss_announce(A:npt.NDArray) -> dict[str,int]:
    """`det_gauss` announcer.
    
    See also
    --------
    - [`det_gauss`][linalg.gauss.det_gauss]
    """
    N:int = A.shape[0]
    return {
        'pos': 1,
        'neg': 1,
        'sub': N*(N**2-1) // 3,
        'mul': max(N*(N**2+2)//3-1, 0),
        'truediv': N*(N-1) // 2
    }

@visualisable(det_gauss_announce, det_gauss_sanitise)
def det_gauss(A:npt.NDArray, *, progress:Progress) -> Any:
    r"""Return the determinant.
    
    $$
        \det A \qquad \mathbb{K}^{N\times N}\to\mathbb{K}
    $$
    
    Uses Gaussian elimination with complete pivoting.
    
    Don't use with data types that don't divide exactly.
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Square matrix.
    progress : Iterable[str]|bool = False
        Progress visualisation specification.
    
    Returns
    -------
    Any
        Determinant, possibly `int(1)` if `A` has shape $0 \times 0$.
    
    Notes
    -----
    The matrix will be wrapped with [`numpy.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)
    and transformed in-place into an upper triangular matrix
    (columns left of pivot won't be reduced).
    
    Complexity
    ----------
    For a matrix of size $N \times N$ there will be at most
    
    - $1$ scalar affirmation (`pos`) or negation (`neg`),
    - $\frac{N(N^2-1)}{3}$ scalar subtractions (`sub`),
    - $\begin{cases} \frac{N(N^2+2)}{3}-1 & N>0 \\ 0 & N=0 \end{cases}$
    scalar multiplications (`mul`) &
    - $\frac{N(N-1)}{2}$ scalar true divisions (`truediv`).
    
    See also
    --------
    - [`det_gauss_sanitise`][linalg.gauss.det_gauss_sanitise]
    - [`det_gauss_announce`][linalg.gauss.det_gauss_announce]
    
    References
    ----------
    [Wikipedia - Gaussian elimination - Computing determinants](https://en.wikipedia.org/wiki/Gaussian_elimination#Computing_determinants)
    """
    N:int = A.shape[0]
    s:bool = True
    for i in range(N):
        #pivot
        i_max, j_max = \
                np.unravel_index(np.argmax(np.abs(A[i:, i:])), A[i:, i:].shape)
        if not A[i+i_max, i+j_max]: #determinant zero, early exit
            M:int = N - i
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
    return progress.posneg(progress.prod_default(np.diag(A)), s)


def inv_gauss_sanitise(A:npt.ArrayLike) -> tuple[Sequence, Mapping]:
    """`inv_gauss` sanitiser.
    
    See also
    --------
    - [`inv_gauss`][linalg.gauss.inv_gauss]
    """
    A = np.asarray(A)
    if not (A.ndim==2 and A.shape[0]==A.shape[1]):
        raise ValueError('A must be two dimensional and square')
    return [A], {}

def inv_gauss_announce(A:npt.NDArray) -> dict[str,int]:
    """`inv_gauss` announcer.
    
    See also
    --------
    - [`inv_gauss`][linalg.gauss.inv_gauss]
    """
    N:int = A.shape[0]
    return {
        'sub': N**2 * (2*N-2),
        'mul': N**2 * (2*N-2),
        'truediv': 2 * N**2
    }

@visualisable(inv_gauss_announce, inv_gauss_sanitise)
def inv_gauss(A:npt.NDArray, *, progress:Progress) -> npt.NDArray:
    r"""Return the inverse.
    
    $$
        A^{-1} \qquad \mathbb{K}^{N\times N}\to\mathbb{K}^{N\times N}
    $$
    
    Uses Gaussian elimination with complete pivoting.
    
    Don't use with data types that don't divide exactly.
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Square matrix.
    progress : Iterable[str]|bool = False
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
    For a matrix of size $N \times N$ there will be
    
    - $2N^3-2N^2$ scalar subtractions (`sub`),
    - $2N^3-2N^2$ scalar multiplications (`mul`) &
    - $2N^2$ scalar true divisions (`truediv`).
    
    TODO: Still unnecessary operations like `A[i, i]/A[i, i]`.
    
    See also
    --------
    - [`inv_gauss_sanitise`][linalg.gauss.inv_gauss_sanitise]
    - [`inv_gauss_announce`][linalg.gauss.inv_gauss_announce]
    
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
