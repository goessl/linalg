import numpy as np
from linalg.rand import randint, randfr



def test_randr():
    for _ in range(100):
        M, N = randint(2, 10, size=2)
        r = randint(0, min(M, N)+1)
        A = randfr(M, N, r)
        
        assert np.linalg.matrix_rank(A.astype(np.float64)) == r
