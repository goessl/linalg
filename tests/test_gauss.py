import numpy as np
from linalg.gauss import *
from linalg.rand import randint, randf



def test_det_gauss():
    for _ in range(100):
        N = randint(1, 10)
        A = randf((N, N))
        
        prediction = det_gauss(A.copy())
        actual = np.linalg.det(A.astype(np.float64))
        assert np.isclose(float(prediction), actual)


def test_inv_gauss():
    for _ in range(100):
        N = randint(1, 10)
        A = randf((N, N))
        
        A_inv = inv_gauss(A.copy())
        assert np.array_equal(A@A_inv, np.eye(N))
