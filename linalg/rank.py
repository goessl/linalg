"""Low rank operations."""



import numpy as np
from numpy.typing import ArrayLike, NDArray
from .blas2 import matmul
from .gauss import inv_gauss, ref_gauss
from .util import dict_sub, dict_isub
from .progress import Progress, visualisable
from typing import Any, Never



__all__ = (
    'rank_decomp_sanitise', 'rank_decomp_announce', 'rank_decomp_cost',
    'rank_decomp',
    'pinv_sanitise', 'pinv_announce', 'pinv_cost', 'pinv',
    'lstsq_sanitise', 'lstsq_announce', 'lstsq_cost', 'lstsq'
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
    """`rank_decomp` operation cost calculation.
    
    See also
    --------
    - [`rank_decomp`][linalg.rank.rank_decomp]
    - [`ref_gauss_cost`][linalg.gauss.ref_gauss_cost]
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
    """`pinv` operation cost calculation.
    
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
    """`lstsq` operation cost calculation.
    
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
    - [`pinv_sanitise`][linalg.rank.pinv_sanitise]
    - [`pinv_announce`][linalg.rank.pinv_announce]
    - [`pinv_cost`][linalg.rank.pinv_cost]
    
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
