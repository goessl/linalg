import numpy as np
from linalg.leibniz import det_leibniz
from linalg.rand import randint, randf
from linalg.counterwrapper import CounterWrapper, toCounterWrappers
from math import factorial



def test_det_leibniz():
    for _ in range(100):
        N = randint(1, 7)
        A = randf((N, N))
        
        CounterWrapper.counter.clear()
        prediction = det_leibniz(toCounterWrappers(A)).v
        actual = np.linalg.det(A.astype(np.float64))
        assert np.isclose(float(prediction), actual)
        
        assert set(CounterWrapper.counter) <= {'+', '*'}
        assert CounterWrapper.counter['+'] == factorial(N)-1
        assert CounterWrapper.counter['*'] == (N-1)*factorial(N)
        assert CounterWrapper.counter.total() == N*factorial(N)-1
