import numpy as np
from linalg.lu import *
from linalg.rand import randint, randf



def test_LU():
    for _ in range(100):
        M, N = randint(1, 10, size=2)
        A = randf((M, N))
        
        L, U = LU(A.copy())
        assert is_tril(L) and is_triu(U) and np.array_equal(L@U, A)
        assert M<=N and np.all(np.diag(L)==1) \
                or A.shape[0]>A.shape[1] and np.all(np.diag(U)==1)

def test_PLU():
    for _ in range(100):
        M, N = randint(1, 10, size=2)
        A = randf((M, N))
        
        P, L, U = PLU(A.copy())
        assert is_perm(P) and is_tril(L) and is_triu(U) \
                and np.array_equal(P@L@U, A)
        assert np.all(np.diag(L) == 1)

def test_LUQ():
    for _ in range(100):
        M, N = randint(1, 10, size=2)
        A = randf((M, N))
        
        L, U, Q = LUQ(A.copy())
        assert is_tril(L) and is_triu(U) and is_perm(Q) \
                and np.array_equal(L@U@Q, A)
        assert np.all(np.diag(U) == 1)

def test_PLUQ():
    for _ in range(100):
        M, N = randint(1, 10, size=2)
        A = randf((M, N))
        
        P, L, U, Q = PLUQ(A.copy())
        assert is_perm(P) and is_tril(L) and is_triu(U) and is_perm(Q) \
                and np.array_equal(P@L@U@Q, A)
        assert M<=N and np.all(np.diag(L)==1) \
                or M>N and np.all(np.diag(U)==1)
