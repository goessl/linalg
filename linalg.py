import numpy as np
from math import prod
from functools import reduce
from operator import mul


#https://stackoverflow.com/a/54069951
def swap_rows(A, i, j):
    A[[i, j], :] = A[[j, i], :]
def swap_columns(A, i, j):
    A[:, [i, j]] = A[:, [j, i]]
def swap_pivot(A, p, i, j):
    swap_rows(A, p, i)
    swap_columns(A, p, j)

def det_gauss(A, criterion=bool):
    """Return the determinant of A.
    
    Calculates the determinant of A by Gaussian elimination with complete pivoting.
    The matrix will nearly be transformed into a upper triangular matrix (columns left of pivot won't be reduced).
    The elements must support abs (numerical return type such that numpy can compare them), division, multiplication, subtraction, negation & have a zero constructor.
    `criterion` will be given the pivot elements and it must return `True` if a division by the element is stable.
    If no non-zero pivot element is found type(A[i,i])(0) is returned.
    """
    #https://en.wikipedia.org/wiki/Gaussian_elimination#Computing_determinants
    s = True
    for i in range(A.shape[0]):
        #pivot
        i_max, j_max = \
                np.unravel_index(np.argmax(abs(A[i:, i:])), A[i:, i:].shape)
        if not criterion(A[i+i_max, i+j_max]):
            return type(A[i+i_max, i+j_max])(0)
        swap_pivot(A, i, i+i_max, i+j_max)
        s ^= bool(i_max) != bool(j_max)
        #reduce (not left of pivot, these elements will not influence result)
        A[i+1:, i:] -= A[i, i:] * (A[i+1:, i] / A[i, i])[:, np.newaxis]
    
    d = reduce(mul, np.diag(A)) if A.dtype==np.object_ else prod(np.diag(A))
    return d if s else -d


if __name__ == '__main__':
    from fractions import Fraction
    
    def randf(shape=None, precision=1000):
        if shape == None:
            n = np.random.randint(-precision, +precision+1)
            d = np.random.randint(1, precision)
            return Fraction(n, d)
        else:
            a = np.empty(shape, dtype=np.object_)
            for i in np.ndindex(*a.shape):
                a[*i] = randf(precision=precision)
            return a
    
    for _ in range(100):
        N = np.random.randint(1, 10)
        A = randf(shape=(N, N))
        
        actual = np.linalg.det(A.astype(np.float64))
        prediction = det_gauss(A)
        assert np.isclose(float(prediction), actual)



def inv_gauss(A, criterion=bool):
    """Return the inverse of A.
    
    Calculates the inverse of A by Gaussian elimination with complete pivoting.
    The matrix will be transformed into the identity matrix.
    The elements must support abs (numerical return type such that numpy can compare them), division, multiplication & subtraction.
    `criterion` will be given the pivot elements and it must return `True` if a division by the element is stable.
    """
    #https://math.stackexchange.com/a/744213/1170417
    P, Q = np.eye(A.shape[0], dtype=A.dtype), np.eye(A.shape[0], dtype=A.dtype)
    for i in range(A.shape[0]):
        #pivot
        i_max, j_max = \
                np.unravel_index(np.argmax(abs(A[i:, i:])), A[i:, i:].shape)
        if not criterion(A[i+i_max, i+j_max]):
            raise ZeroDivisionError
        swap_pivot(A, i, i+i_max, i+j_max)
        swap_rows(P, i, i+i_max), swap_columns(Q, i, i+j_max)
        #normalize pivot
        P[i, :] /= A[i, i]
        A[i, :] /= A[i, i]
        #zeros above and below
        P[:i, :] -= P[i, :] * A[:i, i][:, np.newaxis]
        A[:i, :] -= A[i, :] * A[:i, i][:, np.newaxis]
        P[i+1:, :] -= P[i, :] * A[i+1:, i][:, np.newaxis]
        A[i+1:, :] -= A[i, :] * A[i+1:, i][:, np.newaxis]
    return Q @ P


if __name__ == '__main__':
    for _ in range(100):
        N = np.random.randint(1, 10)
        A = randf(shape=(N, N))
        
        actual = np.linalg.inv(A.astype(np.float64))
        prediction = inv_gauss(A)
        assert np.allclose(prediction.astype(np.float64), actual)



from operator import eq
from itertools import pairwise

def is_row_echelon(A, reduced=True, comparison=eq):
    #https://stackoverflow.com/a/19502403
    pivots = [next((i for i, x in enumerate(row) if not comparison(x, 0)), A.shape[1]) for row in A]
    #check all pivots go to the right
    if not all(pi<pj or pi==A.shape[1] for pi, pj in pairwise(pivots)):
        return False
    
    if reduced:
        #check if pivots one and zeros above
        for i, p in enumerate(pivots):
            if p<A.shape[1]:
                if not comparison(A[i, p], 1) or any(not comparison(A[k, p], 0) for k in range(0, i)):
                    return False
    return True

def row_echelon(A, reduced=True, criterion=bool):
    """Transform A into (reduced) row echelon form and return the pivot column indices.
    
    Transform A into (reduced) row echelon form by Gaussian elimination with pivoting.
    The transformation happens in-place.
    The elements must support abs (numerical return type such that numpy can compare them), division, multiplication & subtraction.
    `criterion` will be given the pivot elements and it must return `True` if a division by the element is stable.
    """
    #https://en.wikipedia.org/wiki/Gaussian_elimination#Pseudocode
    i, j = 0, 0
    pivots = []
    while i<A.shape[0] and j<A.shape[1]:
        #find pivot
        if not criterion(A[(i_max := np.argmax(abs(A[i:, j])) + i), j]):
            j += 1
        else:
            #pivot
            swap_rows(A, i, i_max)
            if reduced:
                #normalize pivot
                A[i, :] /= A[i, j]
                #zeros above and below
                A[:i, :] -= A[i, :] * A[:i, j][:, np.newaxis]
                A[i+1:, :] -= A[i, :] * A[i+1:, j][:, np.newaxis]
            else:
                #zeros below
                A[i+1:, :] -= A[i, :] * (A[i+1:, j] / A[i, j])[:, np.newaxis]
            pivots += [j]
            i += 1
            j += 1
    return pivots


if __name__ == '__main__':
    for _ in range(100):
        N = np.random.randint(1, 10)
        A = randf(shape=(N, N))
        row_echelon(A, reduced=False)
        assert is_row_echelon(A, reduced=False)
        
        A = randf(shape=(N, N))
        row_echelon(A, reduced=True)
        assert is_row_echelon(A, reduced=True)



from scipy.stats import ortho_group

def rands(M, N, s):
    """Return a MxN matrix with the given singular values."""
    S = np.zeros((M, N))
    for i, si in enumerate(s):
        S[i, i] = si
    U, V = ortho_group.rvs(M), ortho_group.rvs(N)
    return U @ S @ V.T

def randr(M, N, r):
    """Return a MxN matrix with rank r.
    
    The singular values are uniformly distributed in ]-1, -0.5]u[+0.5, +1[."""
    s = (np.random.random(size=r)/2 + 0.5) * np.random.choice((-1, +1), size=r)
    s = np.pad(s, (0, min(M, N)-r))
    return rands(M, N, s)

def rank_decomposition(A, criterion=bool):
    """Return a rank decomposition B, C of A such that A=BC.
    
    `criterion` will be given elements and it must return `True` if a division by the element is stable.
    """
    #https://en.wikipedia.org/wiki/Rank_factorization#Rank_factorization_from_reduced_row_echelon_forms
    pivots = row_echelon(B:=A.copy(), criterion=criterion)
    #delete non pivot columns from A
    A = np.delete(A, [i for i in range(A.shape[1]) if i not in pivots], 1)
    #delete zero rows from B
    for i in reversed(range(B.shape[0])):
        if all(not criterion(B[i, j]) for j in range(B.shape[1])):
            B = np.delete(B, i, 0)
    return A, B


if __name__ == '__main__':
    for _ in range(100):
        M, N = np.random.randint(2, 10, size=2)
        s = sorted(np.random.random(min(M, N)), reverse=True)
        A = rands(M, N, s)
        
        assert A.shape==(M, N) and np.allclose(sorted(s, reverse=True),
                np.linalg.svd(A, compute_uv=False))
    
    for _ in range(100):
        M, N = np.random.randint(2, 10, size=2)
        r = np.random.randint(0, min(M, N)+1)
        A = randr(M, N, r)
        
        assert A.shape==(M, N) and np.linalg.matrix_rank(A)==r
    
    for _ in range(100):
        M, N  = np.random.randint(2, 10, size=2)
        r = np.random.randint(0, min(M, N)+1)
        A = randr(M, N, r)
        B, C = rank_decomposition(A, criterion=lambda x: not np.isclose(x, 0))
        
        assert B.shape==(M, r) and C.shape==(r, N)
        assert np.allclose(A, B@C)



def pseudo_inverse(A, criterion=bool):
    #https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Rank_decomposition
    B, C = rank_decomposition(A, criterion=criterion)
    return C.T @ inv_gauss(C@C.T, criterion=criterion) \
            @ inv_gauss(B.T@B, criterion=criterion) @ B.T


if __name__ == '__main__':
    for _ in range(100):
        M, N = np.random.randint(2, 10, size=2)
        r = np.random.randint(0, min(M, N)+1)
        A = randr(M, N, r)
        
        Ainv = pseudo_inverse(A, criterion=lambda x: not np.isclose(x, 0))
        #https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Definition
        assert np.allclose(A@Ainv@A, A) \
                and np.allclose(Ainv@A@Ainv, Ainv) \
                and np.allclose((A@Ainv).conj().T, A@Ainv) \
                and np.allclose((Ainv@A).conj().T, Ainv@A)



def is_tril(L, criterion=bool):
    return all(not criterion(Li) for Li in L[np.triu_indices_from(L, k=1)])

def is_triu(U, criterion=bool):
    return all(not criterion(Ui) for Ui in U[np.tril_indices_from(U, k=-1)])

def is_perm(P):
    #https://en.wikipedia.org/wiki/Permutation_matrix
    return np.array_equal(P.T@P, np.eye(P.shape[0])) and np.all(P>=0)

def LU(A, criterion=bool):
    """Return the LU decomposition of A.
    
    Transformation happens in-place (A becomes U).
    Doolittle decomposition (L diagonal ones) if `A.shape[0]<=A.shape[1]`,
    Crout decomposition (U diagonal ones) otherwise.
    https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
    Elements must support division, multiplication & subtraction.
    """
    A = A if (T:=A.shape[1]>=A.shape[0]) else A.T #more size efficient
    L = np.identity(A.shape[0], dtype=A.dtype)
    
    for i in range(A.shape[0]-1):
        if not criterion(A[i, i]):
            raise ZeroDivisionError
        L[i+1:, i] = A[i+1:, i] / A[i, i]
        A[i+1:,:] -= L[i+1:, i][:, np.newaxis] * A[i, :]
    return (L, A) if T else (A.T, L.T)

def PLU(A, criterion=bool):
    """Return the PLU decomposition of A.
    
    Transformation happens in-place (A becomes U).
    Doolittle decomposition (L diagonal ones).
    https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
    Memory efficient if `A.shape[0]<=A.shape[1]`,
    otherwise LUQ would be more efficient.
    Elements must support abs (numerical return type such that
    numpy can compare them), division, multiplication & subtraction.
    """
    P = np.identity(A.shape[0], dtype=np.int_)
    L = np.identity(A.shape[0], dtype=A.dtype)
    
    for i in range(min(A.shape[0]-1, A.shape[1])):
        if not criterion(A[(p := np.argmax(abs(A[i:, i])) + i), i]):
            raise ZeroDivisionError
        swap_columns(P, i, p)
        swap_pivot(L, i, p, p)
        swap_rows(A, i, p)
        
        L[i+1:, i] = A[i+1:, i] / A[i, i]
        A[i+1:, :] -= L[i+1:, i][:, np.newaxis] * A[i, :]
    return P, L, A

def LUQ(A, criterion=bool):
    """Return the LUQ decomposition of A.
    
    Transformation happens in-place (A becomes L).
    Crout decomposition (U diagonal ones).
    https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
    Memory efficient if `A.shape[0]>A.shape[1]`,
    otherwise LUQ would be more efficient.
    Elements must support abs (numerical return type such that
    numpy can compare them), division, multiplication & subtraction.
    """
    U = np.identity(A.shape[1], dtype=A.dtype)
    Q = np.identity(A.shape[1], dtype=np.int_)
    
    for i in range(min(A.shape[0], A.shape[1]-1)):
        if not criterion(A[i, (p := np.argmax(abs(A[i, i:])) + i)]):
            raise ZeroDivisionError
        swap_columns(A, i, p)
        swap_pivot(U, i, p, p)
        swap_rows(Q, i, p)
        
        U[i, i+1:] = A[i, i+1:] / A[i, i]
        A[:, i+1:] -= U[i, i+1:] * A[:, i][:, np.newaxis]
    return A, U, Q

def PLUQ(A, criterion=bool):
    """Return the PLUQ decomposition of A.
    
    Transformation happens in-place (A becomes U).
    Doolittle decomposition (L diagonal ones) if `A.shape[0]<=A.shape[1]`,
    Crout decomposition (U diagonal ones) otherwise.
    https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
    Elements must support abs (numerical return type such that
    numpy can compare them), division, multiplication & subtraction.
    """
    if A.shape[0] > A.shape[1]:
        P, L, U, Q = PLUQ(A.T, criterion=criterion)
        return Q.T, U.T, L.T, P.T
    
    P = np.identity(A.shape[0], dtype=np.int_)
    L = np.identity(A.shape[0], dtype=A.dtype)
    Q = np.identity(A.shape[1], dtype=np.int_)
    
    for i in range(A.shape[0]-1):
        p_row, p_col = \
                np.unravel_index(np.argmax(abs(A[i:, i:])), A[i:, i:].shape)
        p_row += i
        p_col += i
        if not criterion(A[p_row, p_col]):
            raise ZeroDivisionError
        swap_columns(P, i, p_row)
        swap_pivot(L, i, p_row, p_row)
        swap_pivot(A, i, p_row, p_col)
        swap_rows(Q, i, p_col)
        
        L[i+1:, i] = A[i+1:, i] / A[i, i]
        A[i+1:, :] -= L[i+1:, i][:, np.newaxis] * A[i, :]
    return P, L, A, Q


if __name__ == '__main__':
    for _ in range(100):
        M, N = np.random.randint(1, 10, size=2)
        A = randf(shape=(M, N))
        
        L, U = LU(A.copy())
        assert is_tril(L) and is_triu(U) and np.array_equal(L@U, A)
        assert A.shape[0]<=A.shape[1] and all(Lii==1 for Lii in np.diag(L)) \
                or A.shape[0]>A.shape[1] and all(Uii==1 for Uii in np.diag(U))
    
    for _ in range(100):
        M, N = np.random.randint(1, 3, size=2)
        A = randf(shape=(M, N))
        
        P, L, U = PLU(A.copy())
        assert is_perm(P) and is_tril(L) and is_triu(U) \
                and np.array_equal(P@L@U, A)
        assert all(Lii==1 for Lii in np.diag(L))
    
    for _ in range(100):
        M, N = np.random.randint(1, 10, size=2)
        A = randf(shape=(M, N))
        
        L, U, Q = LUQ(A.copy())
        assert is_tril(L) and is_triu(U) and is_perm(Q) \
                and np.array_equal(L@U@Q, A)
        assert all(Uii==1 for Uii in np.diag(U))
    
    for _ in range(100):
        M, N = np.random.randint(1, 10, size=2)
        A = randf(shape=(M, N))
        
        P, L, U, Q = PLUQ(A.copy())
        assert is_perm(P) and is_tril(L) and is_triu(U) and is_perm(Q) \
                and np.array_equal(P@L@U@Q, A)
        assert A.shape[0]<=A.shape[1] and all(Lii==1 for Lii in np.diag(L)) \
                or A.shape[0]>A.shape[1] and all(Uii==1 for Uii in np.diag(U))
