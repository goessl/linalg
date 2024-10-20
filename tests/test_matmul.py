import numpy as np
from linalg.matmul import matmul
from linalg.rand import randint, randf
from linalg.counterwrapper import *



def test_matmul():
    for _ in range(100):
        L, N, M = randint(1, 20, size=3)
        A, B = randf((L, N)), randf((N, M))
        
        CounterWrapper.counter.clear()
        prediction = fromCounterWrappers(
                matmul(toCounterWrappers(A), toCounterWrappers(B)))
        actual = A @ B
        assert np.array_equal(prediction, actual)
        
        assert set(CounterWrapper.counter) <= {'+', '*'}
        assert CounterWrapper.counter['+'] == L*(N-1)*M
        assert CounterWrapper.counter['*'] == L*N*M
        assert CounterWrapper.counter.total() == L*(2*N-1)*M
