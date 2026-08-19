"""Low rank operations."""



import numpy as np
from numpy.typing import ArrayLike, NDArray
from .gauss import ref_gauss_announce, ref_gauss
from .progress import Progress, visualisable
from typing import Never



__all__ = (
    'rank_decomp_sanitise', 'rank_decomp_announce', 'rank_decomp'
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
    return [A], {}

def rank_decomp_announce(A: NDArray) -> dict[str,int]:
    """`rank_decomp` announcer.
    
    See also
    --------
    - [`rank_decomp`][linalg.rank.rank_decomp]
    - [`ref_gauss_announce`][linalg.gauss.ref_gauss_announce]
    """
    return ref_gauss_announce(A, reduced=True)

@visualisable(rank_decomp_announce, rank_decomp_sanitise)
def rank_decomp(A: ArrayLike, *, progress: Progress) -> tuple[NDArray,NDArray]:
    r"""Return a rank decomposition $B, C$ of $A$ such that $A=BC$.
    
    $$
        B, C \qquad \mathbb{K}^{M\times N}\to\mathbb{K}^{M\times R}\times\mathbb{K}^{R\times N} \quad \operatorname{rank}A=R
    $$
    
    Parameters
    ----------
    A : numpy.typing.ArrayLike
    progress : Iterable[str]|bool = False
        Progress visualisation specification.
    
    Returns
    -------
    B, C : numpy.typing.NDArray
    
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
    
    References
    ----------
    [Wikipedia - Rank factorization - Rank factorization from reduced row echelon forms](https://en.wikipedia.org/wiki/Rank_factorization#Rank_factorization_from_reduced_row_echelon_forms)
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
