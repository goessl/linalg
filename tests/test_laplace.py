import numpy as np
from linalg.laplace import det_laplace
from linalg.rand import randint, randf



def test_det_laplace():
    for _ in range(100):
        N = randint(1, 6)
        A = randf((N, N))
        
        prediction = det_laplace(A)
        actual = np.linalg.det(A.astype(np.float64))
        assert np.isclose(float(prediction), actual)
