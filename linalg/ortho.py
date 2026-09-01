"""Orthogonalisations and normalisations.

TODO: make [`gram_schmidt`][linalg.ortho.gram_schmidt]
a wrapper of ['qr_decomp'][linalg.ortho.qr_decomp] or vise versa.
"""



from itertools import combinations, combinations_with_replacement
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .blas import vsub, vmul, vtruediv
from .blas2 import matmul
from .progress import visualisable, Progress
from typing import Any
from collections.abc import Callable



__all__ = (
    'are_orthogonal', 'are_normalised', 'are_orthonormal',
    'is_orthogonal', 'is_semiorthogonal',
    'is_orthonormal', 'is_semiorthonormal',
    'gram_schmidt_sanitise', 'gram_schmidt_cost', 'gram_schmidt_announce',
    'gram_schmidt',
    'qr_decomp_sanitise', 'qr_decomp_cost', 'qr_decomp_announce', 'qr_decomp'
)



def are_orthogonal(vs: ArrayLike) -> bool:
    r"""Return whether all row vectors are orthogonal.
    
    $$
        \vec{v}_i\overset{?}{\perp}\vec{v}_j \qquad \forall i \neq j
    $$
    
    TODO: Type safe zero check.
    
    Parameters
    ----------
    vs : numpy.typing.ArrayLike
        Row vectors.
    
    Returns
    -------
    bool
        Whether all vectors are orthogonal.
    
    See also
    --------
    - [`are_orthonormal`][linalg.ortho.are_orthonormal]
    """
    vs = np.asarray(vs)
    if vs.ndim != 2:
        raise ValueError('vs must be two dimensional')
    
    return all(not bool(v@w) for v, w in combinations(vs, 2))

def are_normalised(vs: ArrayLike) -> bool:
    r"""Return whether a vector or all row vectors are normalised.
    
    $$
        \vec{v}_i\cdot\vec{v}_i\overset{?}{=}1 \qquad \forall i
    $$
    
    TODO: Type safe zero/one check.
    
    Parameters
    ----------
    vs : numpy.typing.ArrayLike
        Single vector or row vectors.
    
    Returns
    -------
    bool
        Whether it or all are normalised.
    
    See also
    --------
    - [`are_orthonormal`][linalg.ortho.are_orthonormal]
    """
    vs = np.asarray(vs)
    if vs.ndim not in {1, 2}:
        raise ValueError('vs must be one or two dimensional')
    
    return vs@vs==1 if vs.ndim==1 else all(v@v==1 for v in vs)

def are_orthonormal(vs: ArrayLike) -> bool:
    r"""Return whether all vectors are orthonormal.
    
    $$
        \vec{v}_i\cdot\vec{v}_j\overset{?}{=}\delta_{ij} \qquad \forall i, j
    $$
    
    TODO: Type safe zero/one check.
    
    Parameters
    ----------
    vs : numpy.typing.ArrayLike
        Row vectors.
    
    Returns
    -------
    bool
        Whether all vectors are orthonormal.
    
    See also
    --------
    - [`are_orthogonal`][linalg.ortho.are_orthogonal]
    - [`are_normalised`][linalg.ortho.are_normalised]
    """
    vs = np.asarray(vs)
    if vs.ndim != 2:
        raise ValueError('vs must be two dimensional')
    
    return all(vs[i]@vs[j]==int(i==j)
               for i, j in combinations_with_replacement(range(len(vs)), 2))


def is_orthogonal(Q: ArrayLike) -> bool:
    r"""Return whether the matrix is orthogonal.
    
    $$
        Q^TQ \ \text{diagonal?}
    $$
    
    TODO: Type safe zero check.
    
    Checks for orthogonal columns, **not** orthonormal columns.
    
    Parameters
    ----------
    Q : numpy.typing.ArrayLike
        Square matrix.
    
    Returns
    -------
    bool
        Whether it is orthogonal.
    
    See also
    --------
    - [`is_orthonormal`][linalg.ortho.is_orthonormal]
    
    References
    ----------
    - [Wikipedia - Orthogonal matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix)
    """
    Q = np.asarray(Q)
    if not (Q.ndim==2 and Q.shape[0]==Q.shape[1]):
        raise ValueError('Q must be two dimensional and square')
    
    return are_orthogonal(Q.T)

def is_semiorthogonal(Q: ArrayLike) -> bool:
    r"""Return whether the matrix is semi-orthogonal.
    
    $$
        Q^TQ \ \text{diagonal} \ \lor QQ^T \ \text{diagonal?}
    $$
    
    TODO: Type safe zero check.
    
    Checks for orthogonal rows/columns, **not** orthonormal rows/columns.
    
    Parameters
    ----------
    Q : numpy.typing.ArrayLike
        Matrix.
    
    Returns
    -------
    bool
        Whether it is semi-orthogonal.
    
    See also
    --------
    - [`is_orthogonal`][linalg.ortho.is_orthogonal]
    
    References
    ----------
    - [Wikipedia - Semi-orthogonal matrix](https://en.wikipedia.org/wiki/Semi-orthogonal_matrix)
    """
    Q = np.asarray(Q)
    if Q.ndim != 2:
        raise ValueError('Q must be two dimensional')
    
    return are_orthogonal(Q.T) or are_orthogonal(Q)

def is_orthonormal(Q: ArrayLike) -> bool:
    r"""Return whether the matrix is orthonormal.
    
    $$
        Q^TQ \overset{?}{=} 1
    $$
    
    TODO: Type safe zero/one check.
    
    Parameters
    ----------
    Q : numpy.typing.ArrayLike
        Square matrix.
    
    Returns
    -------
    bool
        Whether it is orthonormal.
    
    See also
    --------
    - [`is_orthogonal`][linalg.ortho.is_orthogonal]
    
    References
    ----------
    - [Wikipedia - Orthogonal matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix)
    """
    Q = np.asarray(Q)
    if not (Q.ndim==2 and Q.shape[0]==Q.shape[1]):
        raise ValueError('Q must be two dimensional and square')
    
    return are_orthonormal(Q)

def is_semiorthonormal(Q: ArrayLike) -> bool:
    r"""Return whether the matrix is semi-orthonormal.
    
    $$
        Q^TQ\overset{?}{=}1 \ \lor \ QQ^T\overset{?}{=}1
    $$
    
    TODO: Type safe zero/one check.
    
    Parameters
    ----------
    Q : numpy.typing.ArrayLike
        Matrix.
    
    Returns
    -------
    bool
        Whether it is semi-orthonormal.
    
    See also
    --------
    - [`is_orthonormal`][linalg.ortho.is_orthonormal]
    
    References
    ----------
    - [Wikipedia - Semi-orthogonal matrix](https://en.wikipedia.org/wiki/Semi-orthogonal_matrix)
    """
    Q = np.asarray(Q)
    if Q.ndim != 2:
        raise ValueError('Q must be two dimensional')
    
    return are_orthonormal(Q.T) or are_orthonormal(Q)



def gram_schmidt_sanitise[T](vs: NDArray[T], *,
        sqrt: Callable[[T],T]|None=None) \
        -> tuple[tuple[NDArray[T]],dict[str,Any]]:
    """`gram_schmidt` sanitiser.
    
    See also
    --------
    - [`gram_schmidt`][linalg.ortho.gram_schmidt]
    """
    if not isinstance(vs, np.ndarray):
        raise TypeError('vs must be a numpy.ndarray')
    if vs.ndim != 2:
        raise ValueError('vs must be two dimensional')
    return (vs,), {'sqrt':sqrt}

def gram_schmidt_cost(M: int, N: int, normalise: bool=False) -> dict[str,int]:
    """`gram_schmidt` cost.
    
    Parameters
    ----------
    M : int
        Number of vectors.
    N : int
        Dimension.
    normalise : bool = False
        Whether normalisation will be done.
    
    Returns
    -------
    dict[str,int]
        Cost.
    
    See also
    --------
    - [`gram_schmidt`][linalg.ortho.gram_schmidt]
    """
    return {
        'add': M*(M+1)//2 * max(N-1, 0),
        'sub': M*(M-1)*N // 2,
        'mul': M**2 * N,
        'truediv': M*N,
        'sqrt': M
    } if normalise else {
        'add': M*(M+1)//2 * max(N-1, 0),
        'sub': M*(M-1)*N // 2,
        'mul': M**2 * N,
        'truediv': M*(M-1) // 2
    }

def gram_schmidt_announce[T](vs: NDArray, *, sqrt: Callable[[T],T]|None=None) \
        -> dict[str,int]:
    """`gram_schmidt` announcer.
    
    See also
    --------
    - [`gram_schmidt`][linalg.ortho.gram_schmidt]
    """
    return gram_schmidt_cost(*vs.shape, normalise=sqrt is not None)

@visualisable(gram_schmidt_announce, gram_schmidt_sanitise)
def gram_schmidt[T](vs: NDArray[T], *, sqrt:Callable[[T],T]|None,
        progress: Progress) -> list[T]:
    r"""Orthogonalise/-normalise row vectors.
    
    Transformation happens in-place.
    
    Orthogonalisation if `sqrt` isn't provided;
    orthonormalisation if `sqrt` is provided.
    
    Don't use with data types that don't divide exactly.
    
    Parameters
    ----------
    vs : numpy.typing.NDArray[T]
        Row vectors.
    sqrt : Callable[[T],T]|None = None
        Optional sqrt function. Orthogonalisation/-normalisation switch.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    list[T]
        The vector norms squared.
    
    Raises
    ------
    ZeroDivisionError
        If the vectors can't be orthogonalised/-normalised.
    
    Complexity
    ----------
    For $M$ vectors of length $N$ there will be for orthogonalisation
    
    - $\frac{M(M+1)}{2}\max\{N-1,0\}$ scalar additions (`add`),
    - $\frac{M(M-1)N}{2}$ scalar subtractions (`sub`),
    - $M^2N$ scalar multiplications (`mul`) &
    - $\frac{M(M-1)}{2}$ scalar true divisions (`truediv`)
    
    and for orthonormalisation
    
    - $\frac{M(M+1)}{2}\max\{N-1,0\}$ scalar additions (`add`),
    - $\frac{M(M-1)N}{2}$ scalar subtractions (`sub`),
    - $M^2N$ scalar multiplications (`mul`),
    - $MN$ scalar true divisions (`truediv`) &
    - $M$ square root calculations (`sqrt`).
    
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
            #vs[i] -= vs[i] @ vs[j] / (vs[j] @ vs[j]) * vs[j]
            #vs[i] -= vs[i] @ vs[j] * vs[j]
            c = matmul(vs[i,:], vs[j,:], progress=progress)
            if sqrt is None:
                c = progress.truediv(c, dots[j])
            vs[i,:] = vsub(vs[i,:],
                           vmul(c, vs[j,:], progress=progress),
                           progress=progress)
        a = matmul(vs[i,:], vs[i,:], progress=progress)
        if not a:
            raise ZeroDivisionError('not orthogonalisable')
        dots.append(a)
        if sqrt is not None:
            #vs[i] /= sqrt(vs[i] @ vs[i])
            a = sqrt(a)
            progress.update('sqrt')
            vs[i,:] = vtruediv(vs[i,:], a, progress=progress)
    return dots


def qr_decomp_sanitise[T](A: ArrayLike, *, sqrt: Callable[[T],T]|None=None) \
        -> tuple[tuple[NDArray],dict[str,Any]]:
    """`qr_decomp` sanitiser.
    
    See also
    --------
    - [`qr_decomp`][linalg.ortho.qr_decomp]
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be two dimensional')
    return (A,), {'sqrt':sqrt}

def qr_decomp_cost(M: int, N: int, normalise: bool=False) -> dict[str,int]:
    """`qr_decomp` cost.
    
    Parameters
    ----------
    M, N : int
        Dimensions.
    normalise : bool = False
        Whether normalisation will be done.
    
    Returns
    -------
    dict[str,int]
        Cost.
    
    See also
    --------
    - [`qr_decomp`][linalg.ortho.qr_decomp]
    """
    return {
        'add': N*(N+1)//2 * max(M-1, 0),
        'sub': N*(N-1)*M // 2,
        'mul': N**2 * M,
        'truediv': N*M,
        'sqrt': N
    } if normalise else {
        'add': N*(N+1)//2 * max(M-1, 0),
        'sub': N*(N-1)*M // 2,
        'mul': N**2 * M,
        'truediv': N*(N-1) // 2
    }

def qr_decomp_announce[T](A: NDArray[T], *, sqrt: Callable[[T],T]|None=None) \
        -> dict[str,int]:
    """`qr_decomp` announcer.
    
    See also
    --------
    - [`qr_decomp`][linalg.ortho.qr_decomp]
    """
    return qr_decomp_cost(*A.shape, normalise=sqrt is not None)

@visualisable(qr_decomp_announce, qr_decomp_sanitise)
def qr_decomp[T](A: NDArray[T], *, sqrt:Callable[[T],T]|None,
        progress: Progress) -> tuple[NDArray[T],NDArray[T]]:
    r"""Return the QR decomposition.
    
    $$
        Q, R \qquad \mathbb{K}^{M\times N}\to\mathbb{K}^{M\times N}\times\mathbb{K}^{N\times N}
    $$
    
    Uses the Gram-Schmidt process on the columns.
    
    If `sqrt` isn't provided:
    
    $Q$ has orthogonal, **not** orthonormal columns
    and $R$ is upper triangular with $r_{jj}=1$.
    
    If `sqrt` is provided:
    
    $Q$ has orthonormal columns and $R$ is upper triangular.
    
    Transformation happens in-place, $A$ becomes $Q$.
    
    Don't use with data types that don't divide exactly.
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
        Matrix.
    sqrt : Callable[[T],T]|None = None
        Optional sqrt function. Orthogonalisation/-normalisation switch.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    Q : numpy.typing.NDArray[T]
        Orthogonal/-normal factor.
    R : numpy.typing.NDArray[T]
        (Unit) upper triangular factor.
    
    Raises
    ------
    ZeroDivisionError
        If the matrix can't be orthogonalised/-normalised.
    
    Complexity
    ----------
    There will be for orthogonalisation
    
    - $\frac{N(N+1)}{2}\max\{M-1,0\}$ scalar additions (`add`),
    - $\frac{N(N-1)M}{2}$ scalar subtractions (`sub`),
    - $N^2M$ scalar multiplications (`mul`) &
    - $\frac{N(N-1)}{2}$ scalar true divisions (`truediv`)
    
    and for orthonormalisation
    
    - $\frac{N(N+1)}{2}\max\{M-1,0\}$ scalar additions (`add`),
    - $\frac{N(N-1)M}{2}$ scalar subtractions (`sub`),
    - $N^2M$ scalar multiplications (`mul`),
    - $NM$ scalar true divisions (`truediv`) &
    - $N$ square root calculations (`sqrt`).
    
    See also
    --------
    - [`qr_decomp_sanitise`][linalg.ortho.qr_decomp_sanitise]
    - [`qr_decomp_announce`][linalg.ortho.qr_decomp_announce]
    - [`qr_decomp_cost`][linalg.ortho.qr_decomp_cost]
    
    References
    ----------
    - [Wikipedia - QR decomposition - Using the Gram-Schmidt process](https://en.wikipedia.org/wiki/QR_decomposition#Using_the_Gram%E2%80%93Schmidt_process)
    """
    R = np.eye(A.shape[1], dtype=A.dtype)
    dots = []
    for j in range(A.shape[1]):
        for i in range(j):
            #A[:,j] -= A[:,j] @ A[:,i] / (A[:,i] @ A[:,i]) * A[:,i]
            #A[:,j] -= A[:,j] @ A[:,i] * A[:,i]
            R[i, j] = matmul(A[:,j], A[:,i], progress=progress)
            if sqrt is None:
                R[i, j] = progress.truediv(R[i,j], dots[i])
            A[:,j] = vsub(
                    A[:,j],
                    vmul(R[i,j], A[:,i], progress=progress),
                    progress=progress
            )
        a = matmul(A[:,j], A[:,j], progress=progress)
        if not a:
            raise ZeroDivisionError('not orthogonalisable')
        dots.append(a)
        if sqrt is not None:
            #A[:,j] /= sqrt(A[:,j] @ A[:,j])
            R[j, j] = sqrt(a)
            progress.update('sqrt')
            A[:,j] = vtruediv(A[:,j], R[j,j], progress=progress)
    return A, R
