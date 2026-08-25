"""Helper functions."""



import numpy as np
from numpy.typing import ArrayLike, NDArray
from collections.abc import Mapping, MutableMapping



__all__ = (
    'dict_add', 'dict_iadd', 'dict_sub', 'dict_isub',
    'swap_rows', 'swap_columns', 'swap_pivot',
    'is_perm', 'is_tril', 'is_triu'
)



def dict_add[K,V](a: Mapping[K,V], b: Mapping[K,V]) -> dict[K,V]:
    """Return the sum of two `Mapping`s as a `dict`.
    
    Keeps zero valued items.
    
    Parameters
    ----------
    a, b : Mapping
        Summands.
    
    Returns
    -------
    dict
        Sum.
    
    Notes
    -----
    Only exists because `collections.Counter` arithmetic
    removes zero valued items.
    """
    return {k:a.get(k, 0)+b.get(k, 0) for k in a.keys()|b.keys()}

def dict_iadd[M: MutableMapping](a: M, b: Mapping) -> M:
    """Add a `Mapping` to another.
    
    Keeps zero valued items.
    
    Parameters
    ----------
    a : MutableMapping
        Summand. Will be modified.
    b : Mapping
        Summand.
    
    Returns
    -------
    MutableMapping
        First summand.
    
    Notes
    -----
    Only exists because `collections.Counter` arithmetic
    removes zero valued items.
    """
    for k, v in b.items():
        if k in a:
            a[k] += v
        else:
            a[k] = v
    return a

def dict_sub[K,V](a: Mapping[K,V], b: Mapping[K,V]) -> dict[K,V]:
    """Return the difference of two `Mapping`s as a `dict`.
    
    Keeps zero valued items.
    
    Parameters
    ----------
    a : Mapping
        Minuend.
    b : Mapping
        Subtrahend.
    
    Returns
    -------
    dict
        Difference.
    
    Notes
    -----
    Only exists because `collections.Counter` arithmetic
    removes zero valued items.
    """
    return {k:a.get(k, 0)-b.get(k, 0) for k in a.keys()|b.keys()}

def dict_isub[M: MutableMapping](a: M, b: Mapping) -> M:
    """Add a `Mapping` to another.
    
    Keeps zero valued items.
    
    Parameters
    ----------
    a : MutableMapping
        Minuend. Will be modified.
    b : Mapping
        Subtrahend.
    
    Returns
    -------
    MutableMapping
        Minuend. Now the difference.
    
    Notes
    -----
    Only exists because `collections.Counter` arithmetic
    removes zero valued items.
    """
    for k, v in b.items():
        if k in a:
            a[k] -= v
        else:
            a[k] = -v
    return a


def swap_rows(A: NDArray, i: int, j: int) -> None:
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

def swap_columns(A: NDArray, i: int, j: int) -> None:
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

def swap_pivot(A: NDArray, p: int, i: int, j: int) -> None:
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


def is_perm(P: ArrayLike) -> bool:
    """Return if `P` is a permutation matrix.
    
    Parameters
    ----------
    P : numpy.typing.ArrayLike
        Matrix.
    
    Returns
    -------
    bool
        Whether `P` is a permutation matrix.
    """
    P = np.asarray(P)
    if not (P.ndim==2 and P.shape[0]==P.shape[1]):
        raise ValueError('P must be two dimensional and square')
    
    #https://stackoverflow.com/a/28896366
    return np.all(P.sum(axis=0) == 1) and np.all(P.sum(axis=1) == 1) \
            and np.all((P == 1) | (P == 0))

def is_tril(L: ArrayLike) -> bool:
    """Return if `L` is lower triangular.
    
    Parameters
    ----------
    L : numpy.typing.ArrayLike
        Matrix.
    
    Returns
    -------
    bool
        Whether `L` is lower triangular.
    
    References
    ----------
    [`numpy.triu_indices_from`](https://numpy.org/doc/stable/reference/generated/numpy.triu_indices_from.html)
    """
    L = np.asarray(L)
    if L.ndim != 2:
        raise ValueError('L must be two dimensional')
    
    return not np.any(L[np.triu_indices_from(L, k=+1)])

def is_triu(U: ArrayLike) -> bool:
    """Return if `U` is upper triangular.
    
    Parameters
    ----------
    U : numpy.typing.ArrayLike
        Matrix.
    
    Returns
    -------
    bool
        Whether `U` is upper triangular.
    
    References
    ----------
    [`numpy.tril_indices_from`](https://numpy.org/doc/stable/reference/generated/numpy.tril_indices_from.html)
    """
    U = np.asarray(U)
    if U.ndim != 2:
        raise ValueError('U must be two dimensional')
    
    return not np.any(U[np.tril_indices_from(U, k=-1)])
