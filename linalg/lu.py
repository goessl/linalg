import numpy as np
from .util import assert_matrix, swap_rows, swap_columns, swap_pivot



def is_perm(P):
    """Return if `P` is a permutation matrix."""
    #https://stackoverflow.com/a/28896366
    return np.all(P.sum(axis=0) == 1) and np.all(P.sum(axis=1) == 1) \
            and np.all((P == 1) | (P == 0))

def is_tril(L):
    """Return if `L` is lower triangular."""
    return not np.any(L[np.triu_indices_from(L, k=+1)])

def is_triu(U):
    """Return if `U` is upper triangular."""
    return not np.any(U[np.tril_indices_from(U, k=-1)])

#https://www.youtube.com/watch?v=Th1EE-65u44
def LU(A):
    """Return the LU decomposition of `A`.
    
    Transformation happens in-place
    (`A` becomes `U` for Doolittle, `L` for Crout).
    For a MxN-matrix returns a MxM- & MxN-matrix for Doolittle
    or a MxN- & NxN-matrix for Crout (automatically most memory efficient).
    Doolittle decomposition ($l_{ii}=1$) if `M<=N`,
    Crout decomposition ($u_{ii}=1$) otherwise.
    https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
    For a matrix (where `M` is the shorter and `N` the longer side length)
    there will be
    - $M(M-1)N/2$ subtractions (`-`),
    - $M(M-1)N/2$ multiplications (`*`),
    - $M(M-1)/2$ divisions (`/`),
    so `M(M-1)N+M(M-1)/2` operations in total.
    """
    assert_matrix(A)
    A = A if (T:=A.shape[0]<=A.shape[1]) else A.T #more size efficient
    L = np.identity(A.shape[0], dtype=A.dtype)
    
    for i in range(A.shape[0]-1):
        if not A[i, i]:
            raise ZeroDivisionError
        L[i+1:, i] = A[i+1:, i] / A[i, i]
        A[i+1:,:] -= L[i+1:, i][:, np.newaxis] * A[i, :]
    return (L, A) if T else (A.T, L.T)

def PLU(A):
    """Return the PLU decomposition of `A`.
    
    Transformation happens in-place (`A` becomes `U`).
    For a MxN-matrix returns a MxM-, MxM- & MxN-matrix.
    Doolittle decomposition ($L_{ii}=1$).
    https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
    If `M>N` `LUQ` would be more efficient.
    There will be (`O=min(M-1, N)`)
    - $NO(2M-O-1)/2$ subtractions (`-`),
    - $NO(2M-O-1)/2$ multiplications (`*`),
    - $O(2M-O-1)/2$ divisions (`/`),
    so $NO(2M-O-1)+O(2M-O-1)/2$ operations in total.
    """
    assert_matrix(A)
    P = np.identity(A.shape[0], dtype=np.int_)
    L = np.identity(A.shape[0], dtype=A.dtype)
    
    for i in range(min(A.shape[0]-1, A.shape[1])):
        p = np.argmax(abs(A[i:, i])) + i
        if not A[p, i]:
            raise ZeroDivisionError
        swap_columns(P, i, p)
        swap_pivot(L, i, p, p)
        swap_rows(A, i, p)
        
        L[i+1:, i] = A[i+1:, i] / A[i, i]
        A[i+1:, :] -= L[i+1:, i][:, np.newaxis] * A[i, :]
    return P, L, A

def LUQ(A):
    """Return the LUQ decomposition of `A`.
    
    Transformation happens in-place (`A` becomes `L`).
    For a MxN-matrix returns a MxN-, NxN- & NxN-matrix.
    Crout decomposition ($U_{ii}=1$).
    https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
    If `M<N` `PLU` would be more memory efficient.
    For a MxN-matrix (`O=min(M, N-1)`) there will be
    - $MO(2N-O-1)/2$ subtractions (`-`),
    - $MO(2N-O-1)/2$ multiplications (`*`),
    - $O(2N-O-1)/2$ divisions (`/`),
    so $MO(2N-O-1)+O(2N-O-1)/2$ operations in total.
    """
    assert_matrix(A)
    U = np.identity(A.shape[1], dtype=A.dtype)
    Q = np.identity(A.shape[1], dtype=np.int_)
    
    for i in range(min(A.shape[0], A.shape[1]-1)):
        p = np.argmax(abs(A[i, i:])) + i
        if not A[i, p]:
            raise ZeroDivisionError
        swap_columns(A, i, p)
        swap_pivot(U, i, p, p)
        swap_rows(Q, i, p)
        
        U[i, i+1:] = A[i, i+1:] / A[i, i]
        A[:, i+1:] -= U[i, i+1:] * A[:, i][:, np.newaxis]
    return A, U, Q

def PLUQ(A):
    """Return the PLUQ decomposition of `A`.
    
    Transformation happens in-place
    (`A` becomes `U` for Doolittle, `L` for Crout).
    For a MxN-matrix returns a MxM-, MxM-, MxN- & NxN-matrix for Doolittle
    or a MxM-, MxN-, NxN & NxN-matrix for Crout
    (automatically most memory efficient).
    Doolittle decomposition ($l_{ii}=1$) if `M<=N`,
    Crout decomposition ($u_{ii}=1$) otherwise.
    https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
    For a matrix (where `M` is the shorter and `N` the longer side length)
    there will be
    - $M(M-1)N/2$ subtractions (`-`),
    - $M(M-1)N/2$ multiplications (`*`),
    - $M(M-1)/2$ divisions (`/`),
    so $M(M-1)N+M(M-1)/2$ operations in total.
    """
    assert_matrix(A)
    if A.shape[0] > A.shape[1]:
        P, L, U, Q = PLUQ(A.T)
        return Q.T, U.T, L.T, P.T
    
    P = np.identity(A.shape[0], dtype=np.int_)
    L = np.identity(A.shape[0], dtype=A.dtype)
    Q = np.identity(A.shape[1], dtype=np.int_)
    
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
        
        L[i+1:, i] = A[i+1:, i] / A[i, i]
        A[i+1:, :] -= L[i+1:, i][:, np.newaxis] * A[i, :]
    return P, L, A, Q
