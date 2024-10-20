import numpy as np
from linalg.laplace import det_laplace
from linalg.rand import randint, randf
from linalg.counterwrapper import CounterWrapper, toCounterWrappers
from math import e, floor, factorial



def test_det_laplace():
    for _ in range(100):
        N = randint(1, 6)
        A = randf((N, N))
        
        CounterWrapper.counter.clear()
        prediction = det_laplace(toCounterWrappers(A)).v
        actual = np.linalg.det(A.astype(np.float64))
        assert np.isclose(float(prediction), actual)
        
        assert set(CounterWrapper.counter) <= {'+', '*'}
        assert CounterWrapper.counter['+'] <= factorial(N)-1
        assert CounterWrapper.counter['*'] <= floor((e-1)*factorial(N))-1
        assert CounterWrapper.counter.total() <= floor(e*factorial(N))-1
