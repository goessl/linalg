"""Triangular decompositions."""



import numpy as np
from numpy.typing import ArrayLike, NDArray
from .blas import vsub, vmul, vtruediv
from .util import swap_rows, swap_columns, swap_pivot
from .progress import visualisable, Progress
from typing import Never



__all__ = (
    'lu_sanitise', 'lu_cost', 'lu_announce', 'lu',
    'plu_sanitise', 'plu_cost', 'plu_announce', 'plu',
    'luq_sanitise', 'luq_cost', 'luq_announce', 'luq',
    'pluq_sanitise', 'pluq_cost', 'pluq_announce', 'pluq'
)



def lu_sanitise(A: ArrayLike) -> tuple[tuple[NDArray],dict[Never,Never]]:
    """`lu` sanitiser.
    
    See also
    --------
    - [`lu`][linalg.triangular.lu]
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be two dimensional')
    return (A,), {}

def lu_cost(M: int, N: int) -> dict[str,int]:
    """`lu` cost.
    
    Parameters
    ----------
    N, M : int
        Dimensions.
    
    Returns
    -------
    dict[str,int]
        Cost.
    
    See also
    --------
    - [`lu`][linalg.triangular.lu]
    """
    M, N = sorted((M, N))
    return {
        'sub': M*(M-1)*N//2,
        'mul': M*(M-1)*N//2,
        'truediv': M*(M-1)//2
    }

def lu_announce(A: NDArray) -> dict[str,int]:
    """`lu` announcer.
    
    See also
    --------
    - [`lu`][linalg.triangular.lu]
    """
    return lu_cost(*A.shape)

@visualisable(lu_announce, lu_sanitise)
def lu[T](A: NDArray[T], *, progress: Progress) \
        -> tuple[NDArray[T],NDArray[T]]:
    r"""Return the LU decomposition.
    
    $$
        L, U \qquad A=LU
    $$
    
    Transformation happens in-place.
    
    For $M \leq N$:
    
    - Doolittle decomposition
    - $l_{ii}=1$
    - $A$ becomes $U$
    - $L\in\mathbb{K}^{M\times M}, \ U\in\mathbb{K}^{M\times N}$
    
    For $M > N$:
    
    - Crout decomposition
    - $u_{ii}=1$
    - $A$ becomes $L$
    - $L\in\mathbb{K}^{M\times N}, \ U\in\mathbb{K}^{N\times N}$
    
    (automatically most memory efficient).
    
    Don't use with data types that don't divide exactly.
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Matrix.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    L : numpy.typing.NDArray
        Lower triangular factor.
    U : numpy.typing.NDArray
        Upper triangular factor.
    
    Throws
    ------
    ZeroDivisionError
        When a zero pivot can not be permutated away.
    
    Complexity
    ----------
    With $m=\min\{M, N\}, \ n=\max\{M, N\}$ there will be
    
    - $m(m-1)n/2$ scalar subtractions (`sub`),
    - $m(m-1)n/2$ scalar multiplications (`mul`) &
    - $m(m-1)/2$ scalar true divisions (`truediv`).
    
    See also
    --------
    - [`lu_sanitise`][linalg.triangular.lu_sanitise]
    - [`lu_announce`][linalg.triangular.lu_announce]
    - [`lu_cost`][linalg.triangular.lu_cost]
    
    References
    ----------
    - [Wikipedia - LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)
    - [Wikipedia - Crout matrix decomposition](https://en.wikipedia.org/wiki/Crout_matrix_decomposition)
    """
    A = A if (T:=A.shape[0]<=A.shape[1]) else A.T #more size efficient
    L = np.identity(A.shape[0], dtype=A.dtype)
    
    for i in range(A.shape[0]-1):
        if not A[i, i]:
            raise ZeroDivisionError
        #L[i+1:, i] = A[i+1:, i] / A[i, i]
        L[i+1:, i] = vtruediv(A[i+1:, i], A[i, i], progress=progress)
        #A[i+1:,:] -= L[i+1:, i][:, np.newaxis] * A[i, :]
        A[i+1:,:] = vsub(A[i+1:,:],
                         vmul(L[i+1:, i][:, np.newaxis],
                              A[i, :],
                              progress=progress),
                         progress=progress)
    return (L, A) if T else (A.T, L.T)


def plu_sanitise(A: ArrayLike) -> tuple[tuple[NDArray],dict[Never,Never]]:
    """`plu` sanitiser.
    
    See also
    --------
    - [`plu`][linalg.triangular.plu]
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be two dimensional')
    return (A,), {}

def plu_cost(M: int, N: int) -> dict[str,int]:
    """`plu` cost.
    
    Parameters
    ----------
    N, M : int
        Dimensions.
    
    Returns
    -------
    dict[str,int]
        Cost.
    
    See also
    --------
    - [`plu`][linalg.triangular.plu]
    """
    O = min(M-1, N)
    return {
        'sub': N*O*(2*M-O-1)//2,
        'mul': N*O*(2*M-O-1)//2,
        'truediv': O*(2*M-O-1)//2
    }

def plu_announce(A: NDArray) -> dict[str,int]:
    """`plu` announcer.
    
    See also
    --------
    - [`plu`][linalg.triangular.plu]
    """
    return plu_cost(*A.shape)

@visualisable(plu_announce, plu_sanitise)
def plu[T](A: NDArray[T], *, progress:Progress) \
        -> tuple[NDArray[bool],NDArray[T],NDArray[T]]:
    r"""Return the PLU decomposition.
    
    $$
        P, L, U \qquad A=PLU
    $$
    
    Transformation happens in-place.
    
    - Doolittle decomposition
    - $l_{ii}=1$
    - $A$ becomes $U$
    - $P\in\mathbb{B}^{M\times M}, \ L\in\mathbb{K}^{M\times M}, \ U\in\mathbb{K}^{M\times N}$
    
    For $M>N$ [`luq`][linalg.triangular.luq] would be more memory efficient.
    
    Don't use with data types that don't divide exactly.
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Matrix.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    P : numpy.typing.NDArray[bool]
        Permutation factor.
    L : numpy.typing.NDArray
        Lower triangular factor.
    U : numpy.typing.NDArray
        Upper triangular factor.
    
    Throws
    ------
    ZeroDivisionError
        When a zero pivot can not be permutated away.
    
    Complexity
    ----------
    With $O=\min\{M-1, N\}$ there will be
    
    - $NO(2M-O-1)/2$ scalar subtractions (`sub`),
    - $NO(2M-O-1)/2$ scalar multiplications (`mul`) &
    - $O(2M-O-1)/2$ scalar true divisions (`truediv`).
    
    See also
    --------
    - [`plu_sanitise`][linalg.triangular.plu_sanitise]
    - [`plu_announce`][linalg.triangular.plu_announce]
    - [`plu_cost`][linalg.triangular.plu_cost]
    
    References
    ----------
    - [Wikipedia - LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)
    - [Wikipedia - Crout matrix decomposition](https://en.wikipedia.org/wiki/Crout_matrix_decomposition)
    """
    P = np.identity(A.shape[0], dtype=bool)
    L = np.identity(A.shape[0], dtype=A.dtype)
    
    for i in range(min(A.shape[0]-1, A.shape[1])):
        p = np.argmax(abs(A[i:, i])) + i
        if not A[p, i]:
            raise ZeroDivisionError
        swap_columns(P, i, p)
        swap_pivot(L, i, p, p)
        swap_rows(A, i, p)
        
        #L[i+1:, i] = A[i+1:, i] / A[i, i]
        L[i+1:, i] = vtruediv(A[i+1:, i], A[i, i], progress=progress)
        #A[i+1:, :] -= L[i+1:, i][:, np.newaxis] * A[i, :]
        A[i+1:, :] = vsub(A[i+1:, :],
                          vmul(L[i+1:, i][:, np.newaxis],
                               A[i, :],
                               progress=progress),
                          progress=progress)
    return P, L, A


def luq_sanitise(A: ArrayLike) -> tuple[tuple[NDArray],dict[Never,Never]]:
    """`luq` sanitiser.
    
    See also
    --------
    - [`luq`][linalg.triangular.luq]
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be two dimensional')
    return (A,), {}

def luq_cost(M: int, N: int) -> dict[str,int]:
    """`luq` cost.
    
    Parameters
    ----------
    N, M : int
        Dimensions.
    
    Returns
    -------
    dict[str,int]
        Cost.
    
    See also
    --------
    - [`luq`][linalg.triangular.luq]
    """
    O = min(M, N-1)
    return {
        'sub': M*O*(2*N-O-1)//2,
        'mul': M*O*(2*N-O-1)//2,
        'truediv': O*(2*N-O-1)//2
    }

def luq_announce(A: NDArray) -> dict[str,int]:
    """`luq` announcer.
    
    See also
    --------
    - [`luq`][linalg.triangular.luq]
    """
    return luq_cost(*A.shape)

@visualisable(luq_announce, luq_sanitise)
def luq[T](A: NDArray[T], *, progress:Progress) \
        -> tuple[NDArray[T],NDArray[T],NDArray[bool]]:
    r"""Return the LUQ decomposition.
    
    $$
        L, U, Q \qquad A=LUQ
    $$
    
    Transformation happens in-place.
    
    - Crout decomposition
    - $u_{ii}=1$
    - $A$ becomes $L$
    - $L\in\mathbb{K}^{M\times N}, \ U\in\mathbb{K}^{N\times N}, \ Q\in\mathbb{B}^{N\times N}$
    
    For $M<N$ [`plu`][linalg.triangular.plu] would be more memory efficient.
    
    Don't use with data types that don't divide exactly.
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Matrix.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    L : numpy.typing.NDArray
        Lower triangular factor.
    U : numpy.typing.NDArray
        Upper triangular factor.
    Q : numpy.typing.NDArray[bool]
        Permutation factor.
    
    Throws
    ------
    ZeroDivisionError
        When a zero pivot can not be permutated away.
    
    Complexity
    ----------
    With $O=\min\{M, N-1\}$ there will be
    
    - $MO(2N-O-1)/2$ scalar subtractions (`sub`),
    - $MO(2N-O-1)/2$ scalar multiplications (`mul`) &
    - $O(2N-O-1)/2$ scalar true divisions (`truediv`).
    
    See also
    --------
    - [`luq_sanitise`][linalg.triangular.luq_sanitise]
    - [`luq_announce`][linalg.triangular.luq_announce]
    - [`luq_cost`][linalg.triangular.luq_cost]
    
    References
    ----------
    - [Wikipedia - LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)
    - [Wikipedia - Crout matrix decomposition](https://en.wikipedia.org/wiki/Crout_matrix_decomposition)
    """
    U = np.identity(A.shape[1], dtype=A.dtype)
    Q = np.identity(A.shape[1], dtype=bool)
    
    for i in range(min(A.shape[0], A.shape[1]-1)):
        p = np.argmax(abs(A[i, i:])) + i
        if not A[i, p]:
            raise ZeroDivisionError
        swap_columns(A, i, p)
        swap_pivot(U, i, p, p)
        swap_rows(Q, i, p)
        
        #U[i, i+1:] = A[i, i+1:] / A[i, i]
        U[i, i+1:] = vtruediv(A[i, i+1:], A[i, i], progress=progress)
        #A[:, i+1:] -= U[i, i+1:] * A[:, i][:, np.newaxis]
        A[:, i+1:] = vsub(A[:, i+1:],
                          vmul(U[i, i+1:],
                               A[:, i][:, np.newaxis],
                               progress=progress),
                          progress=progress)
    return A, U, Q


def pluq_sanitise(A: ArrayLike) -> tuple[tuple[NDArray],dict[Never,Never]]:
    """`pluq` sanitiser.
    
    See also
    --------
    - [`pluq`][linalg.triangular.pluq]
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be two dimensional')
    return (A,), {}

def pluq_cost(M: int, N: int) -> dict[str,int]:
    """`pluq` cost.
    
    Parameters
    ----------
    N, M : int
        Dimensions.
    
    Returns
    -------
    dict[str,int]
        Cost.
    
    See also
    --------
    - [`pluq`][linalg.triangular.pluq]
    """
    M, N = sorted((M, N))
    return {
        'sub': M*(M-1)*N//2,
        'mul': M*(M-1)*N//2,
        'truediv': M*(M-1)//2
    }

def pluq_announce(A: NDArray) -> dict[str,int]:
    """`pluq` announcer.
    
    See also
    --------
    - [`pluq`][linalg.triangular.pluq]
    """
    return pluq_cost(*A.shape)

@visualisable(pluq_announce, pluq_sanitise)
def pluq[T](A: NDArray[T], *, progress: Progress) \
        -> tuple[NDArray[bool],NDArray[T],NDArray[T],NDArray[bool]]:
    r"""Return the PLUQ decomposition.
    
    $$
        P, L, U, Q \qquad A=PLUQ
    $$
    
    Transformation happens in-place.
    
    For $M \leq N$:
    
    - Doolittle decomposition
    - $l_{ii}=1$
    - $A$ becomes $U$
    - $P\in\mathbb{B}^{M\times M}, \ L\in\mathbb{K}^{M\times M}, \ U\in\mathbb{K}^{M\times N}, \ Q\in\mathbb{B}^{N\times N}$
    
    For $M > N$:
    
    - Crout decomposition
    - $u_{ii}=1$
    - $A$ becomes $L$
    - $P\in\mathbb{B}^{M\times M}, \ L\in\mathbb{K}^{M\times N}, \ U\in\mathbb{K}^{N\times N}, \ Q\in\mathbb{B}^{N\times N}$
    
    (automatically most memory efficient).
    
    Don't use with data types that don't divide exactly.
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Matrix.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    P : numpy.typing.NDArray[bool]
        Permutation factor.
    L : numpy.typing.NDArray
        Lower triangular factor.
    U : numpy.typing.NDArray
        Upper triangular factor.
    Q : numpy.typing.NDArray[bool]
        Permutation factor.
    
    Complexity
    ----------
    With $m=\min\{M, N\}, \ n=\max\{M, N\}$ there will be
    
    - $m(m-1)n/2$ scalar subtractions (`sub`),
    - $m(m-1)n/2$ scalar multiplications (`mul`) &
    - $m(m-1)/2$ scalar true divisions (`truediv`).
    
    See also
    --------
    - [`pluq_sanitise`][linalg.triangular.pluq_sanitise]
    - [`pluq_announce`][linalg.triangular.pluq_announce]
    - [`pluq_cost`][linalg.triangular.pluq_cost]
    
    References
    ----------
    - [Wikipedia - LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition)
    - [Wikipedia - Crout matrix decomposition](https://en.wikipedia.org/wiki/Crout_matrix_decomposition)
    """
    if A.shape[0] > A.shape[1]:
        P, L, U, Q = pluq(A.T, progress=progress)
        return Q.T, U.T, L.T, P.T
    
    P = np.identity(A.shape[0], dtype=bool)
    L = np.identity(A.shape[0], dtype=A.dtype)
    Q = np.identity(A.shape[1], dtype=bool)
    
    for i in range(A.shape[0]-1):
        p_row, p_col = \
                np.unravel_index(np.argmax(abs(A[i:, i:])), A[i:, i:].shape)
        p_row += i
        p_col += i
        if not A[p_row, p_col]:
            raise ZeroDivisionError
        swap_columns(P, i, p_row)
        swap_pivot(L, i, p_row, p_row)
        swap_pivot(A, i, p_row, p_col)
        swap_rows(Q, i, p_col)
        
        #L[i+1:, i] = A[i+1:, i] / A[i, i]
        L[i+1:, i] = vtruediv(A[i+1:, i], A[i, i], progress=progress)
        #A[i+1:, :] -= L[i+1:, i][:, np.newaxis] * A[i, :]
        A[i+1:, :] = vsub(A[i+1:, :],
                          vmul(L[i+1:, i][:, np.newaxis],
                               A[i, :],
                               progress=progress),
                          progress=progress)
    return P, L, A, Q
