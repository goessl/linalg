"""Low rank operations."""



import numpy as np
from numpy.typing import ArrayLike, NDArray
from iteration import MISSING
from .blas2 import matmul
from .gauss import inv_gauss, ref_gauss
from .util import dict_sub, dict_isub
from .progress import Progress, visualisable
from typing import Any, Never



__all__ = (
    'rank_decomp_sanitise', 'rank_decomp_cost', 'rank_decomp_announce',
    'rank_decomp',
    'nullspace_sanitise', 'nullspace_cost', 'nullspace_announce', 'nullspace',
    'pinv_sanitise', 'pinv_cost', 'pinv_announce', 'pinv',
    'lstsq_sanitise', 'lstsq_cost', 'lstsq_announce', 'lstsq'
)



def rank_decomp_sanitise(A: ArrayLike) \
        -> tuple[tuple[NDArray],dict[Never,Never]]:
    """`rank_decomp` sanitiser.
    
    See also
    --------
    - [`rank_decomp`][linalg.rank.rank_decomp]
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be two dimensional')
    return (A,), {}

def rank_decomp_cost(M: int, N: int, R: int|None=None) -> dict[str,int]:
    """`rank_decomp` cost.
    
    Parameters
    ----------
    N, M : int
        Dimensions.
    R : int
        Rank.
    
    Returns
    -------
    dict[str,int]
        Cost.
    
    See also
    --------
    - [`rank_decomp`][linalg.rank.rank_decomp]
    """
    if R is None:
        R = min(M, N)
    return {
        'sub': N*R*(M-1),
        'mul': N*R*(M-1),
        'truediv': N*R
    }

def rank_decomp_announce(A: NDArray) -> dict[str,int]:
    """`rank_decomp` announcer.
    
    See also
    --------
    - [`rank_decomp`][linalg.rank.rank_decomp]
    - [`ref_gauss_announce`][linalg.gauss.ref_gauss_announce]
    """
    return rank_decomp_cost(*A.shape)

@visualisable(rank_decomp_announce, rank_decomp_sanitise)
def rank_decomp[T](A: NDArray[T], *, progress: Progress) \
        -> tuple[NDArray[T],NDArray[T]]:
    r"""Return the rank decomposition $B, C$ of $A$ such that $A=BC$.
    
    $$
        B, C \qquad \mathbb{K}^{M\times N}\to\mathbb{K}^{M\times R}\times\mathbb{K}^{R\times N} \quad \operatorname{rank}A=R
    $$
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Matrix.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    B : numpy.typing.NDArray
        First factor.
    C : numpy.typing.NDArray
        Second factor.
    
    Complexity
    ----------
    There will be
    
    - $NR(M-1)$ scalar subtractions (`sub`),
    - $NR(M-1)$ scalar multiplications (`mul`) &
    - $NR$ scalar true divisions (`truediv`).
    
    See also
    --------
    - [`rank_decomp_sanitise`][linalg.rank.rank_decomp_sanitise]
    - [`rank_decomp_announce`][linalg.rank.rank_decomp_announce]
    - [`rank_decomp_cost`][linalg.rank.rank_decomp_cost]
    
    References
    ----------
    - [Wikipedia - Rank factorization - Rank factorization from reduced row echelon forms](https://en.wikipedia.org/wiki/Rank_factorization#Rank_factorization_from_reduced_row_echelon_forms)
    """
    C = A.copy()
    pivots = ref_gauss(C, progress=progress)
    #delete non pivot columns from A
    B = np.delete(A, [i for i in range(A.shape[1]) if i not in pivots], 1)
    #delete zero rows from C
    for i in reversed(range(C.shape[0])):
        if np.all(np.logical_not(C[i, :])):
            C = np.delete(C, i, 0)
    return B, C


def nullspace_sanitise(A: ArrayLike, *, zero: Any=MISSING, one: Any=MISSING) \
        -> tuple[tuple[NDArray],dict[str,Any]]:
    """`nullspace` sanitiser.
    
    See also
    --------
    - [`nullspace`][linalg.rank.nullspace]
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be two dimensional')
    
    #don't use .item(), would unpack the numpy type to a Python type
    if zero is MISSING:
        zero = np.zeros((), dtype=A.dtype)[()]
    if one is MISSING:
        one = np.ones((), dtype=A.dtype)[()]
    
    return (A,), {'zero':zero, 'one':one}

def nullspace_cost(M: int, N: int, R: int|None=None) -> dict[str,int]:
    """`nullspace` cost.
    
    Parameters
    ----------
    M, N : int
        Dimensions.
    R : int|None = None
        Rank.
    
    Returns
    -------
    dict[str,int]
        Cost.
    
    Notes
    -----
    Full rank is **not** the worst case:
    the elimination gets more expensive with every pivot,
    but the $R(N-R)$ negations peak at $R=N/2$.
    
    See also
    --------
    - [`nullspace`][linalg.rank.nullspace]
    """
    if R is None:
        R, S = min(M, N), min(M, N, N//2)
    else:
        S = R
    return {
        'sub': N*R*(M-1),
        'mul': N*R*(M-1),
        'truediv': N*R,
        'neg': S*(N-S)
    }

def nullspace_announce(A: NDArray, *, zero: Any=0, one: Any=1) \
        -> dict[str,int]:
    """`nullspace` announcer.
    
    See also
    --------
    - [`nullspace`][linalg.rank.nullspace]
    """
    return nullspace_cost(*A.shape)

@visualisable(nullspace_announce, nullspace_sanitise)
def nullspace[T,U,V](A: NDArray[T], *, zero: U, one: V, progress: Progress) \
        -> NDArray[T|U|V]:
    r"""Return a basis of the nullspace (kernel).
    
    $$
        K \quad AK=0 \qquad \mathbb{K}^{M\times N}\to\mathbb{K}^{N\times(N-R)} \quad \operatorname{rank}A=R
    $$
    
    TODO: Review. Currently written by Claude.
    
    The basis vectors are the **columns** of the returned matrix.
    
    Reduces `A` to reduced row echelon form and reads the free columns off
    it. The basis is the usual one, so it is neither orthogonal nor
    normalised.
    
    Don't use with data types that don't divide exactly.
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Matrix.
    zero : Any = MISSING
        Zero element.
    one : Any = MISSING
        One element.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    numpy.typing.NDArray
        Nullspace basis in the columns.
    
    Edge cases
    ----------
    For an object matrix provide `zero` & `one` arguments to not get type
    unspecific `int(0)` & `int(1)` entries back.
    
    A full column rank matrix gives a $N \times 0$ matrix back,
    a $0 \times N$ one the $N \times N$ identity (the whole space).
    
    Complexity
    ----------
    With $R=\operatorname{rank}A$ there will be
    
    - $NR(M-1)$ scalar subtractions (`sub`),
    - $NR(M-1)$ scalar multiplications (`mul`),
    - $NR$ scalar true divisions (`truediv`) &
    - $R(N-R)$ scalar negations (`neg`).
    
    Notes
    -----
    The matrix will be wrapped with [`numpy.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html)
    and transformed in-place into its reduced row echelon form.
    
    The zero and one entries are filled in as the very same objects, not as
    copies, like [`numpy.full`](https://numpy.org/doc/stable/reference/generated/numpy.full.html)
    does. Use immutable scalars.
    
    See also
    --------
    - [`nullspace_sanitise`][linalg.rank.nullspace_sanitise]
    - [`nullspace_announce`][linalg.rank.nullspace_announce]
    - [`nullspace_cost`][linalg.rank.nullspace_cost]
    
    References
    ----------
    - [Wikipedia - Kernel (linear algebra) - Computation by Gaussian elimination](https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Computation_by_Gaussian_elimination)
    """
    pivots = ref_gauss(A, progress=progress)
    
    #progress adjustment, `ref_gauss` corrects its own share internally,
    #only the negations, that depend on the actual rank, are left
    M, N = A.shape
    R, S = len(pivots), min(M, N, N//2)
    progress.update('neg', S*(N-S) - R*(N-R))
    
    free = [j for j in range(N) if j not in pivots]
    K = np.full((N, len(free)), zero, dtype=A.dtype)
    for k, j in enumerate(free):
        K[j, k] = one
        #the pivot variables follow from the free one set to one
        for i, p in enumerate(pivots):
            K[p, k] = progress.neg(A[i, j])
    return K


def pinv_sanitise(A: ArrayLike, *, zero: Any=0) \
        -> tuple[tuple[NDArray],dict[str,Any]]:
    """`pinv` sanitiser.
    
    See also
    --------
    - [`pinv`][linalg.rank.pinv]
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be two dimensional')
    return (A,), {'zero':zero}

def pinv_cost(M: int, N: int, R: int|None=None) -> dict[str,int]:
    """`pinv` cost.
    
    Parameters
    ----------
    N, M : int
        Dimensions.
    R : int
        Rank.
    
    Returns
    -------
    dict[str,int]
        Cost.
    
    See also
    --------
    - [`pinv`][linalg.rank.pinv]
    """
    R = min(M, N) if R is None else R
    L = min(M, N)
    return {
        'add':     R**3 + R*R*(M+N+L-3) - L*R + M*N*max(R-1, 0),
        'sub':     N*R*(M-1) + 4*R*R*(R-1),
        'mul':     5*R**3 + R*R*(M+N+L-4) + N*R*(2*M-1),
        'truediv': R*(N + 4*R)
    }

def pinv_announce(A: NDArray, *, zero: Any=0) -> dict[str,int]:
    """`pinv` announcer.
    
    See also
    --------
    - [`pinv`][linalg.rank.pinv]
    """
    return pinv_cost(*A.shape)

@visualisable(pinv_announce, pinv_sanitise)
def pinv[T,U](A: NDArray[T], *, zero: U, progress: Progress) -> NDArray[T|U]:
    r"""Return the Moore–Penrose pseudo inverse.
    
    $$
        A^+ \qquad \mathbb{K}^{M\times N}\to\mathbb{K}^{N\times M}
    $$
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Matrix.
    zero : Any = 0
        Zero element.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    numpy.typing.NDArray
        Pseudo inverse.
    
    Complexity
    ----------
    With $R=\operatorname{rank}A$ and $L=\min\{M, N\}$ there will be
    
    - $R^3+R^2(M+N+L-3)-LR+MN\max\{R-1,0\}$ scalar additions (`add`),
    - $NR(M-1)+4R^2(R-1)$ scalar subtractions (`sub`),
    - $5R^3+R^2(M+N+L-4)+NR(2M-1)$ scalar multiplications (`mul`) &
    - $R(N+4R)$ scalar true divisions (`truediv`).
    
    Notes
    -----
    TODO: document multiplication order
    
    $$
        \underbrace{C^T}_{N \times R}\underbrace{(CC^T)^{-1}}_{R \times R}\underbrace{(B^TB)^{-1}}_{R \times R}\underbrace{B^T}_{R \times M}
    $$
    
    See also
    --------
    - [`pinv_sanitise`][linalg.rank.pinv_sanitise]
    - [`pinv_announce`][linalg.rank.pinv_announce]
    - [`pinv_cost`][linalg.rank.pinv_cost]
    
    References
    ----------
    - [Moore-Penrose inverse - Rank decomposition](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Rank_decomposition)
    """
    B, C = rank_decomp(A, progress=progress)
    
    #progress adjustment
    predicted = pinv_cost(*A.shape)
    actual = pinv_cost(*A.shape, B.shape[1])
    adjustment = dict_sub(predicted, actual)
    rank_decomp_adjustment = dict_sub(rank_decomp_cost(*A.shape),
                                      rank_decomp_cost(*A.shape, B.shape[1]))
    dict_isub(adjustment, rank_decomp_adjustment)
    for k, v in adjustment.items():
        progress.update(k, v)
    
    M = matmul(inv_gauss(matmul(C, C.T, zero=zero, progress=progress),
                         progress=progress),
               inv_gauss(matmul(B.T, B, zero=zero, progress=progress),
                         progress=progress),
               progress=progress
    )
    if A.shape[1] <= A.shape[0]: #(C.T @ M) @ B.T
        return matmul(matmul(C.T, M, zero=zero, progress=progress),
                      B.T,
                      zero=zero,
                      progress=progress
        )
    else: #C.T @ (M @ B.T)
        return matmul(C.T,
                      matmul(M, B.T, zero=zero, progress=progress),
                      zero=zero,
                      progress=progress
        )


def lstsq_sanitise(X: ArrayLike, y: ArrayLike, *, zero: Any=0) \
        -> tuple[tuple[NDArray, NDArray],dict[str,Any]]:
    """`lstsq` sanitiser.
    
    See also
    --------
    - [`lstsq`][linalg.rank.lstsq]
    """
    X, y = np.asarray(X), np.asarray(y)
    if not(X.ndim==2 and y.ndim==1):
        raise ValueError('X must be two dimensional, y one dimensional')
    if X.shape[0] != y.shape[0]:
        raise ValueError('X & y must have the same height')
    return (X, y), {'zero':zero}

def lstsq_cost(M: int, N: int) -> dict[str,int]:
    """`lstsq` cost.
    
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
    - [`lstsq`][linalg.rank.lstsq]
    """
    return {
        'add':     N*(N+1)*max(M-1, 0) + N*(N-1)*(5*N+1),
        'sub':     5*N*N*(N-1),
        'mul':     M*N*(N+1) + 2*N*N*(5*N-2),
        'truediv': 5*N*N
    }

def lstsq_announce(X: NDArray, y: NDArray, *, zero: Any=0) -> dict[str,int]:
    """`lstsq` announcer.
    
    See also
    --------
    - [`lstsq`][linalg.rank.lstsq]
    """
    return lstsq_cost(*X.shape)

@visualisable(lstsq_announce, lstsq_sanitise)
def lstsq[T,U](X: NDArray[T], y: NDArray[T], *, zero: U, progress: Progress) \
        -> NDArray[T|U]:
    r"""Return the linear least squares solution.
    
    $$
        \vec{b} \quad X\vec{b}=\vec{y} \qquad \mathbb{K}^{M\times N}\times\mathbb{K}^M\to\mathbb{K}^N
    $$
    
    Parameters
    ----------
    X : numpy.typing.ArrayLike
        Inputs. Samples in rows, features in columns.
    y : numpy.typing.ArrayLike
        Outputs.
    zero : Any = 0
        Zero element.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    numpy.typing.NDArray
        Pseudo inverse.
    
    Complexity
    ----------
    With $R=\operatorname{rank}A$ and $L=\min\{M, N\}$ there will be
    
    - $N(N+1)\max\{M-1,0\} + N(N-1)(5N+1)$ scalar additions (`add`),
    - $5N^2(N-1)$ scalar subtractions (`sub`),
    - $MN(N+1) + 2N^2(5N-2)$ scalar multiplications (`mul`) &
    - $5N^2$ scalar true divisions (`truediv`).
    
    Notes
    -----
    TODO: document multiplication order
    
    $$
        \underbrace{(X^TX)^{-1}}_{N \times N}\underbrace{X^T}_{N \times M}\underbrace{\vec{y}}_{M}
    $$
    
    See also
    --------
    - [`lstsq_sanitise`][linalg.rank.lstsq_sanitise]
    - [`lstsq_announce`][linalg.rank.lstsq_announce]
    - [`lstsq_cost`][linalg.rank.lstsq_cost]
    
    References
    ----------
    - [Moore-Penrose inverse - Rank decomposition](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Rank_decomposition)
    """
    XTX_inv = pinv(matmul(X.T, X, zero=zero, progress=progress),
                   zero=zero, progress=progress)
    return matmul(XTX_inv,
                  matmul(X.T, y, zero=zero, progress=progress),
                  zero=zero,
                  progress=progress
    )
