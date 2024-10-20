import numpy as np
from linalg.lu import *
from linalg.rand import randint, randf
from linalg.counterwrapper import *



def test_LU():
    for _ in range(100):
        M, N = randint(1, 10, size=2)
        A = randf((M, N))
        
        try:
            CounterWrapper.counter.clear()
            L, U = map(fromCounterWrappers, LU(toCounterWrappers(A)))
            assert is_tril(L) and is_triu(U) and np.array_equal(L@U, A)
            assert M<=N and np.all(np.diag(L)==1) \
                    or A.shape[0]>A.shape[1] and np.all(np.diag(U)==1)
            
            assert set(CounterWrapper.counter) <= {'+', '*', '/'}
            M, N = sorted((M, N))
            assert CounterWrapper.counter['+'] == M*(M-1)*N//2
            assert CounterWrapper.counter['*'] == M*(M-1)*N//2
            assert CounterWrapper.counter['/'] == M*(M-1)//2
            assert CounterWrapper.counter.total() == M*(M-1)*N+M*(M-1)//2
        
        except ZeroDivisionError:
            pass


def test_PLU():
    for _ in range(100):
        M, N = randint(1, 10, size=2)
        A = randf((M, N))
        
        try:
            CounterWrapper.counter.clear()
            P, L, U = map(fromCounterWrappers, PLU(toCounterWrappers(A)))
            assert is_perm(P) and is_tril(L) and is_triu(U) \
                    and np.array_equal(P@L@U, A)
            assert np.all(np.diag(L) == 1)
            
            assert set(CounterWrapper.counter) <= {'+', '*', '/'}
            O = min(M-1, N)
            assert CounterWrapper.counter['+'] == N*O*(2*M-O-1)//2
            assert CounterWrapper.counter['*'] == N*O*(2*M-O-1)//2
            assert CounterWrapper.counter['/'] == O*(2*M-O-1)//2
            assert CounterWrapper.counter.total() == N*O*(2*M-O-1)+O*(2*M-O-1)//2
        
        except ZeroDivisionError:
            pass


def test_LUQ():
    for _ in range(100):
        M, N = randint(1, 10, size=2)
        A = randf((M, N))
        
        try:
            CounterWrapper.counter.clear()
            L, U, Q = map(fromCounterWrappers, LUQ(toCounterWrappers(A)))
            assert is_tril(L) and is_triu(U) and is_perm(Q) \
                    and np.array_equal(L@U@Q, A)
            assert np.all(np.diag(U) == 1)
            
            assert set(CounterWrapper.counter) <= {'+', '*', '/'}
            O = min(M, N-1)
            assert CounterWrapper.counter['+'] == M*O*(2*N-O-1)//2
            assert CounterWrapper.counter['*'] == M*O*(2*N-O-1)//2
            assert CounterWrapper.counter['/'] == O*(2*N-O-1)//2
            assert CounterWrapper.counter.total() == M*O*(2*N-O-1)+O*(2*N-O-1)//2
        
        except ZeroDivisionError:
            pass


def test_PLUQ():
    for _ in range(100):
        M, N = randint(1, 10, size=2)
        A = randf((M, N))
        
        try:
            CounterWrapper.counter.clear()
            P, L, U, Q = map(fromCounterWrappers, PLUQ(toCounterWrappers(A)))
            assert is_perm(P) and is_tril(L) and is_triu(U) and is_perm(Q) \
                    and np.array_equal(P@L@U@Q, A)
            assert M<=N and np.all(np.diag(L)==1) \
                    or M>N and np.all(np.diag(U)==1)
            
            assert set(CounterWrapper.counter) <= {'+', '*', '/'}
            M, N = sorted((M, N))
            assert CounterWrapper.counter['+'] == M*N*(M-1)//2
            assert CounterWrapper.counter['*'] == M*N*(M-1)//2
            assert CounterWrapper.counter['/'] == M*(M-1)//2
            assert CounterWrapper.counter.total() == M*N*(M-1)+M*(M-1)//2
        
        except ZeroDivisionError:
            pass
