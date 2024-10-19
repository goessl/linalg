import numpy as np
from linalg.bareiss import det_bareiss
from linalg.rand import randint



def test_det_bareiss():
    for _ in range(1000):
        N = randint(1, 10)
        A = randint(-100, +100, size=(N, N))
        
        prediction = det_bareiss(A.copy())
        actual = np.linalg.det(A.astype(np.int64))
        assert isinstance(prediction, int) \
                and np.isclose(float(prediction), actual)
