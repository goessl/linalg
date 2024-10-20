import numpy as np
from linalg.bareiss import det_bareiss
from linalg.rand import randint
from linalg.counterwrapper import CounterWrapper, toCounterWrappers



def test_det_bareiss():
    for _ in range(1000):
        N = randint(1, 10)
        A = randint(-100, +100, size=(N, N))
        
        CounterWrapper.counter.clear()
        prediction = det_bareiss(toCounterWrappers(A)) #might be int(0)
        prediction = prediction.v if isinstance(prediction, CounterWrapper) \
                else prediction
        actual = np.linalg.det(A.astype(np.int64))
        assert isinstance(prediction, int) \
                and np.isclose(float(prediction), actual)
        
        assert set(CounterWrapper.counter) <= {'-', '*', '//'}
        assert CounterWrapper.counter['-'] == N*(N**2-1)//3
        assert CounterWrapper.counter['*'] == 2*N*(N**2-1)//3
        assert CounterWrapper.counter['//'] == N*(N**2-3*N+2)//3
        assert CounterWrapper.counter.total() == N*(4*N**2-3*N-1)//3
