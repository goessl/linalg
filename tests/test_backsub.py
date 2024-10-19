import numpy as np
from linalg.backsub import backsub
from linalg.rand import randint, randf



def test_backsub():
    for _ in range(100):
        N = randint(1, 10)
        L = np.tril(randf((N, N)))
        x = randf(N)
        try:
            assert np.array_equal(x, backsub(L, L@x, tril=True))
        except:
            assert np.any(np.diag(L) == 0)
        
        U = np.triu(randf((N, N)))
        try:
            assert np.array_equal(x, backsub(U, U@x))
        except:
            assert np.any(np.diag(U) == 0)
