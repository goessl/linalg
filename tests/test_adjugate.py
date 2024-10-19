import numpy as np
from linalg.adjugate import adj
from linalg.gauss import det_gauss
from linalg.rand import randint, randf



def test_adjugate():
    for _ in range(100):
        N = randint(2, 6)
        A = randf((N, N))
        
        adjA = adj(A)
        assert np.array_equal(A@adjA, det_gauss(A)*np.eye(N, dtype=int))
