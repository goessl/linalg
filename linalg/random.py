"""Random sampling."""



from random import binomialvariate
from fractions import Fraction
import numpy as np
from numpy.typing import NDArray



__all__ = (
    'randz', 'randq',
    'vrandz', 'vrandq',
    'mrandqr'
)



def randz(sigma: int=1000) -> int:
    r"""Return a random binomial distributed integer.
    
    $$
        x \qquad x\in\mathbb{Z}, \ x\sim\mathcal{N}(0,\sigma)
    $$
    
    Parameters
    ----------
    sigma : int = 1000
        Standard deviation.
    
    Returns
    -------
    int
        Random value.
    
    See also
    --------
    [`vrandz`][linalg.random.vrandz]
    """
    if sigma < 0:
        raise ValueError('sigma must be non-negative')
    return binomialvariate(4*sigma**2) - 2*sigma**2

def randq(grade: int=1000) -> Fraction:
    r"""Return a random binomial distributed rational.
    
    $$
        x \qquad x\in\mathbb{Q}, \ x\sim\mathcal{N}(0,1)
    $$
    
    Parameters
    ----------
    grade : int = 1000
        Denominator grading.
    
    Returns
    -------
    fractions.Fraction
        Random value.
    
    See also
    --------
    [`vrandq`][linalg.random.vrandq]
    """
    if grade <= 0:
        raise ValueError('grade must be positive')
    return Fraction(randz(grade), grade)


def vrandz(shape: int|tuple[int,...]=1, sigma: int=1000) -> NDArray[object]:
    r"""Return an array of normally distributed integers.
    
    $$
        \mathbb{Z}^\text{shape}
    $$
    
    Just an array filled with [`randz`][linalg.random.randz].
    
    Parameters
    ----------
    shape : int|tuple[int,...] = 1
        Shape.
    sigma : int = 1000
        Standard deviation.
    
    Returns
    -------
    NDArray[object]
        Random sample.
    
    See also
    --------
    [`randz`][linalg.random.randz]
    """
    r = np.empty(shape, dtype=object)
    for i in np.ndindex(r.shape):
        r[i] = randz(sigma)
    return r

def vrandq(shape: int|tuple[int,...]=1, grade: int=1000) -> NDArray[object]:
    r"""Return an array of normaly distributed rationals.
    
    $$
        \mathbb{Q}^\text{shape}
    $$
    
    Just an array filled with [`randq`][linalg.random.randq].
    
    Parameters
    ----------
    shape : int|tuple[int,...] = 1
        Shape.
    grade : int = 1000
        Denominator grading.
    
    Returns
    -------
    NDArray[object]
        Random sample.
    
    See also
    --------
    [`randq`][linalg.random.randq]
    """
    r = np.empty(shape, dtype=object)
    for i in np.ndindex(r.shape):
        r[i] = randq(grade)
    return r


def mrandqr(M: int, N: int, R: int=0, grade: int=1000) \
        -> NDArray[object]:
    r"""Return a rational matrix of specific rank.
    
    $$
        \mathbb{Q}^{M\times N} \qquad \text{with rank $R$}
    $$
    
    Parameters
    ----------
    M, N : int
        Shape.
    R : int
        Rank.
    grade : int = 1000
        Denominator grading.
    
    Returns
    -------
    NDArray[object]
        Random sample.
    
    See also
    --------
    [`randq`][linalg.random.randq]
    [`vrandq`][linalg.random.vrandq]
    """
    if not R: #an empty matmul would fill with int(0)
        return np.full((M, N), Fraction(0), dtype=object)
    return vrandq((M, R), grade=grade) @ vrandq((R, N), grade=grade)