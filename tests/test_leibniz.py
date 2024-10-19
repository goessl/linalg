import numpy as np
from linalg.leibniz import det_leibniz
from linalg.rand import randint, randf



def test_det_leibniz():
    for _ in range(100):
        N = randint(1, 7)
        A = randf((N, N))
        
        prediction = det_leibniz(A)
        actual = np.linalg.det(A.astype(np.float64))
        assert np.isclose(float(prediction), actual)
