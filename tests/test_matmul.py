import numpy as np
from linalg.matmul import matmul
from linalg.rand import randint, randf



def test_matmul():
    for _ in range(100):
        L, N, M = randint(1, 20, size=3)
        A, B = randf((L, N)), randf((N, M))
        
        prediction = matmul(A, B)
        actual = A @ B
        assert np.array_equal(prediction, actual)
