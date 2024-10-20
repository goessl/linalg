import numpy as np
from linalg.gauss import *
from linalg.rand import randint, randf
from linalg.counterwrapper import *



def test_det_gauss():
    for _ in range(100):
        N = randint(1, 10)
        A = randf((N, N))
        
        CounterWrapper.counter.clear()
        prediction = det_gauss(toCounterWrappers(A)).v
        actual = np.linalg.det(A.astype(np.float64))
        assert np.isclose(float(prediction), actual)
        
        assert set(CounterWrapper.counter) <= {'-', '*', '/'}
        assert CounterWrapper.counter['-'] == N*(N**2-1)//3
        assert CounterWrapper.counter['*'] == N*(N**2+2)//3-1
        assert CounterWrapper.counter['/'] == N*(N-1)//2
        assert CounterWrapper.counter.total() == N*(4*N**2+3*N-1)//6-1


def test_inv_gauss():
    for _ in range(100):
        N = randint(1, 10)
        A = randf((N, N))
        
        CounterWrapper.counter.clear()
        A_inv = fromCounterWrappers(inv_gauss(toCounterWrappers(A)))
        assert np.array_equal(A@A_inv, np.eye(N))
        
        assert set(CounterWrapper.counter) <= {'+', '-', '*', '/'}
        assert CounterWrapper.counter['+'] == N**3-N**2
        assert CounterWrapper.counter['-'] == 2*N**3-3*N**2+2*N-1
        assert CounterWrapper.counter['*'] == 2*N**3-2*N**2
        assert CounterWrapper.counter['/'] == 2*N**2-N+1
        assert CounterWrapper.counter.total() == 5*N**3-4*N**2+N
