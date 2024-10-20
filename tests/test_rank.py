import numpy as np
from linalg.rank import *
from linalg.rand import *
from linalg.counterwrapper import *



def test_ref_gauss():
    for _ in range(100):
        M, N  = randint(1, 10, size=2)
        r = randint(0, min(M, N)+1)
        A = randfr(M, N, r)
        
        CounterWrapper.counter.clear()
        A = toCounterWrappers(A)
        ref_gauss(A, False)
        A = fromCounterWrappers(A)
        assert is_ref(A, False)
        
        assert set(CounterWrapper.counter) <= {'-', '*', '/'}
        assert CounterWrapper.counter['-'] == N*r*(2*M-r-1)//2
        assert CounterWrapper.counter['*'] == N*r*(2*M-r-1)//2
        assert CounterWrapper.counter['/'] == r*(2*M-r-1)//2
        assert CounterWrapper.counter.total() == r*(2*M-r-1)*(2*N+1)//2
        
        
        A = randfr(M, N, r)
        
        CounterWrapper.counter.clear()
        A = toCounterWrappers(A)
        ref_gauss(A, True)
        A = fromCounterWrappers(A)
        assert is_ref(A, True)
        
        assert set(CounterWrapper.counter) <= {'-', '*', '/'}
        assert CounterWrapper.counter['-'] == N*r*(M-1)
        assert CounterWrapper.counter['*'] == N*r*(M-1)
        assert CounterWrapper.counter['/'] == N*r
        assert CounterWrapper.counter.total() == N*r*(2*M-1)


def test_rank_decomposition():
    for _ in range(100):
        M, N  = randint(1, 10, size=2)
        r = randint(0, min(M, N)+1)
        A = randfr(M, N, r)
        B, C = rank_decomposition(A)
        
        assert B.shape==(M, r) and C.shape==(r, N) and np.array_equal(A, B@C)

def test_pinv():
    for _ in range(100):
        M, N = randint(1, 10, size=2)
        r = randint(0, min(M, N)+1)
        A = randfr(M, N, r)
        
        Ainv = pinv(A)
        #https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Definition
        assert np.array_equal(A@Ainv@A, A) \
                and np.array_equal(Ainv@A@Ainv, Ainv) \
                and np.array_equal((A@Ainv).conj().T, A@Ainv) \
                and np.array_equal((Ainv@A).conj().T, Ainv@A)

def test_lstsq():
    for _ in range(100):
        M, N = randint(1, 10, size=2)
        X, y = randf((M, N)), randf(M)
        
        prediction = lstsq(X, y)
        actual = np.linalg.lstsq(
                X.astype(np.float64), y.astype(np.float64), rcond=None)[0]
        assert np.allclose(prediction.astype(np.float64), actual)
