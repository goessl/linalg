import numpy as np
from linalg.backsub import backsub
from linalg.rand import randint, randf
from linalg.counterwrapper import *



def test_backsub():
    for _ in range(100):
        N = randint(1, 10)
        x = randf(N)
        L = np.tril(randf((N, N)))
        Lx = L @ x
        
        try:
            CounterWrapper.counter.clear()
            prediction = fromCounterWrappers(backsub(
                    toCounterWrappers(L), toCounterWrappers(Lx), tril=True))
            assert np.array_equal(prediction, x)
            
            assert set(CounterWrapper.counter) <= {'+', '-', '*', '/'}
            assert CounterWrapper.counter['+'] == N*(N-3)//2+1
            assert CounterWrapper.counter['-'] == N-1
            assert CounterWrapper.counter['*'] == N*(N-1)//2
            assert CounterWrapper.counter['/'] == N
            assert CounterWrapper.counter.total() == N**2
            
        except ZeroDivisionError:
            assert np.any(np.diag(L) == 0)
        
        
        U = np.triu(randf((N, N)))
        Ux = U @ x
        try:
            CounterWrapper.counter.clear()
            prediction = fromCounterWrappers(backsub(
                    toCounterWrappers(U), toCounterWrappers(Ux)))
            assert np.array_equal(prediction, x)
            
            assert set(CounterWrapper.counter) <= {'+', '-', '*', '/'}
            assert CounterWrapper.counter['+'] == N*(N-3)//2+1
            assert CounterWrapper.counter['-'] == N-1
            assert CounterWrapper.counter['*'] == N*(N-1)//2
            assert CounterWrapper.counter['/'] == N
            assert CounterWrapper.counter.total() == N**2
            
        except ZeroDivisionError:
            assert np.any(np.diag(U) == 0)
