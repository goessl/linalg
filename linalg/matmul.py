import numpy as np
from .progress import Progress
from .util import assert_matrix



def matmul(A, B, progress=set()):
    """Return the matrix product of `A` & `B`.
    
    Matrices must be non-empty (L,N,M>0).
    For a LxN- & a NxM-matrix (where the result will be LxM) there will be
    - L(N-1)M additions (`+`),
    - LNM multiplications (`*`),
    - so L(2N-1)M operations in total.
    """
    assert_matrix(A)
    assert_matrix(B)
    assert A.shape[1] == B.shape[0]
    totals = {
        '+': A.shape[0] * (A.shape[1] - 1) * B.shape[1],
        '*': A.shape[0] * A.shape[1] * B.shape[1]
    }
    with Progress({o:totals[o] for o in progress}, 'matmul ') as progress:
        C = np.empty((A.shape[0], B.shape[1]), dtype=np.result_type(A, B))
        for i, j in np.ndindex(*C.shape):
            C[i, j] = progress.sum(progress.mul(aik, bkj)
                    for aik, bkj in zip(A[i, :],  B[:, j]))
        return C
