"""Helper functions."""



import numpy as np
from numpy.typing import NDArray
from collections.abc import Mapping, MutableMapping



__all__ = (
    'dict_add', 'dict_iadd', 'dict_sub', 'dict_isub',
    'swap_rows', 'swap_columns', 'swap_pivot'
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
    - [stackoverflow - Swap two rows in a numpy array in python](https://stackoverflow.com/a/54069951)
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
    - [stackoverflow - Swap two rows in a numpy array in python](https://stackoverflow.com/a/54069951)
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
    - [stackoverflow - Swap two rows in a numpy array in python](https://stackoverflow.com/a/54069951)
    """
    if not isinstance(A, np.ndarray):
        raise TypeError('A must be a numpy.ndarray')
    if not A.ndim == 2:
        raise ValueError('A must be two dimensional')
    
    swap_rows(A, p, i)
    swap_columns(A, p, j)
