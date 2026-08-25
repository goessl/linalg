from linalg.blas2 import *
from linalg.random import vrandq
from fractions import Fraction
from itertools import pairwise
from functools import reduce
import numpy as np
import pytest



@pytest.mark.filterwarnings('error')
def test_matmul_vector_x_vector():
    for N in range(10):
        v, w = np.random.rand(N), np.random.rand(N)
        assert np.isclose(matmul(v, w), v@w)

@pytest.mark.filterwarnings('error')
def test_matmul_vector_x_vector_empty():
    #numpy types
    v, w = np.empty(0, np.int64), np.empty(0, np.float64)
    vw = matmul(v, w)
    assert vw==0 and isinstance(vw, np.float64)
    
    #objects
    v, w = np.empty(0, np.int64), np.empty(0, object)
    vw = matmul(v, w)
    assert vw==0 and isinstance(vw, int)
    
    v, w = np.empty(0, np.int64), np.empty(0, object)
    vw = matmul(v, w, zero=Fraction())
    assert vw==0 and isinstance(vw, Fraction)


@pytest.mark.filterwarnings('error')
def test_matmul_vector_x_matrix():
    for M in range(10):
        for N in range(10):
            v, B = np.random.rand(M), np.random.rand(M, N)
            assert np.allclose(matmul(v, B), v@B)

@pytest.mark.filterwarnings('error')
def test_matmul_vector_x_matrix_empty():
    #numpy types
    v, B = np.empty(10, np.int64), np.empty((10, 0), np.float64)
    vB = matmul(v, B)
    assert vB.shape==(0,) and vB.dtype==np.float64
    
    v, B = np.empty(0, np.int64), np.empty((0, 10), np.float64)
    vB = matmul(v, B)
    assert vB.shape==(10,) and np.allclose(vB, 0) and vB.dtype==np.float64
    
    v, B = np.empty(0, np.int64), np.empty((0, 0), np.float64)
    vB = matmul(v, B)
    assert vB.shape==(0,) and vB.dtype==np.float64
    
    #objects
    v, B = np.empty(0, np.int64), np.empty((0, 10), object)
    vB = matmul(v, B)
    assert all(vB[i]==0 and isinstance(vB[i], int)
               for i in np.ndindex(vB.shape))
    
    v, B = np.empty(0, np.int64), np.empty((0, 10), object)
    vB = matmul(v, B, zero=Fraction())
    assert all(vB[i]==0 and isinstance(vB[i], Fraction)
               for i in np.ndindex(vB.shape))


@pytest.mark.filterwarnings('error')
def test_matmul_matrix_x_vector():
    for L in range(10):
        for M in range(10):
            A, v = np.random.rand(L, M), np.random.rand(M)
            assert np.allclose(matmul(A, v), A@v)

@pytest.mark.filterwarnings('error')
def test_matmul_matrix_x_vector_empty():
    #numpy types
    A, v = np.empty((0, 10), np.int64), np.empty(10, np.float64)
    Av = matmul(A, v)
    assert Av.shape==(0,) and Av.dtype==np.float64
    
    A, v = np.empty((10, 0), np.int64), np.empty(0, np.float64)
    Av = matmul(A, v)
    assert Av.shape==(10,) and np.allclose(Av, 0) and Av.dtype==np.float64
    
    A, v = np.empty((0, 0), np.int64), np.empty(0, np.float64)
    Av = matmul(A, v)
    assert Av.shape==(0,) and Av.dtype==np.float64
    
    #objects
    A, v = np.empty((10, 0), np.int64), np.empty(0, object)
    Av = matmul(A, v)
    assert all(Av[i]==0 and isinstance(Av[i], int)
               for i in np.ndindex(Av.shape))
    
    A, v = np.empty((10, 0), np.int64), np.empty(0, object)
    Av = matmul(A, v, zero=Fraction())
    assert all(Av[i]==0 and isinstance(Av[i], Fraction)
               for i in np.ndindex(Av.shape))


@pytest.mark.filterwarnings('error')
def test_matmul_matrix_x_matrix():
    for L in range(10):
        for M in range(10):
            for N in range(10):
                A, B = np.random.rand(L, M), np.random.rand(M, N)
                assert np.allclose(matmul(A, B), A@B)

@pytest.mark.filterwarnings('error')
def test_matmul_matrix_x_matrix_empty():
    #L, M, N zero combinations already tested in test_matmul_matrix_x_matrix
    #only test empty sums and non empty result array here
    
    #numpy types
    A, B = np.empty((10, 0), np.int64), np.empty((0, 20), np.float64)
    AB = matmul(A, B)
    assert np.array_equal(AB, np.zeros((10, 20))) and AB.dtype==np.float64
    
    A, B = np.empty((0, 0), np.int64), np.empty((0, 0), np.float64)
    AB = matmul(A, B)
    assert np.array_equal(AB, np.zeros((0, 0))) and AB.dtype==np.float64
    
    #objects
    A, B = np.empty((10, 0), np.int64), np.empty((0, 20), object)
    AB = matmul(A, B)
    assert all(AB[i,j]==0 and isinstance(AB[i,j], int)
               for i, j in np.ndindex(AB.shape))
    
    A, B = np.empty((10, 0), np.int64), np.empty((0, 20), object)
    AB = matmul(A, B, zero=Fraction())
    assert all(AB[i,j]==0 and isinstance(AB[i,j], Fraction)
               for i, j in np.ndindex(AB.shape))


@pytest.mark.filterwarnings('error')
def test_outer():
    for M in range(10):
        for N in range(10):
            v, w = np.random.rand(M), np.random.rand(N)
            assert np.allclose(outer(v, w), np.outer(v, w))


@pytest.mark.filterwarnings('error')
def test_matmulchain():
    args, kwargs = matmulchain_sanitise(np.empty((10, 30)),
                                        np.empty((30, 5)),
                                        np.empty((5, 60)))
    assert matmulchain_announce(*args, **kwargs) == {'mul': 4500, 'add': 3850}
    
    s = np.random.randint(0, 10, size=np.random.randint(2, 10))
    matrices = [vrandq((M, N)) for M, N in pairwise(s)]
    assert np.array_equal(
        matmulchain(*matrices),
        reduce(np.matmul, matrices)
    )



#############################################################################
#appended by Claude - counting is always on, so any call here also checks
#its announcer; these cover selection, exact totals & the sanitiser.
#the `bars` fixture lives in conftest.py.
#############################################################################

#every category `Progress` knows, for asserting that the untracked ones
#are rejected rather than silently dropped
CATEGORIES = ('pos', 'neg', 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod')



#promotion
#`matmul` subsumes the former `dot`: a 1-D `A` gets a 1 prepended, a 1-D `B`
#gets a 1 appended, and both are stripped off the result again - the same
#rule `numpy.matmul`/`@` follows.
@pytest.mark.parametrize('sa, sb', [
    ((2, 3), (3, 4)),   #matrix x matrix
    ((2, 3), (3,)),     #matrix x vector
    ((3,),   (3, 4)),   #vector^T x matrix
    ((3,),   (3,)),     #vector^T x vector
])
@pytest.mark.filterwarnings('error')
def test_matmul_promotes_like_numpy(sa, sb):
    A, B = np.random.rand(*sa), np.random.rand(*sb)
    r = matmul(A, B)
    assert np.shape(r) == np.shape(A @ B)
    assert np.allclose(r, A @ B)

def test_matmul_of_two_vectors_returns_a_scalar():
    #the pinned 1s are stripped all the way down, not left as a 0-d array
    r = matmul(np.arange(3.), np.arange(3.))
    assert np.ndim(r) == 0 and not isinstance(r, np.ndarray)



#announcements
@pytest.mark.parametrize('f, v, w', [
    (matmul, np.zeros(3),      np.zeros(3)),
    (matmul, np.zeros((2, 2)), np.zeros((2, 2))),
    (outer,  np.zeros(3),      np.zeros(3)),
])
def test_untracked_category_raises(f, v, w):
    #none of these track 'mod', so naming it is a user mistake
    with pytest.raises(ValueError, match='untracked'):
        f(v, w, progress=CATEGORIES)

def test_outer_does_not_track_add(bars):
    #`outer` announces 'mul' only, so 'add' must be rejected too
    with pytest.raises(ValueError, match='untracked'):
        outer(np.zeros(3), np.zeros(3), progress=('mul', 'add'))

@pytest.mark.parametrize('sa, sb, add, mul', [
    ((2, 3), (3, 4), 16, 24),   #L=2, M=3, N=4
    ((2, 3), (3,),    4,  6),   #L=2, M=3, N=1
    ((3,),   (3, 4),  8, 12),   #L=1, M=3, N=4
    ((3,),   (3,),    2,  3),   #L=1, M=3, N=1
])
def test_matmul_announces_every_rank_combination(bars, sa, sb, add, mul):
    #`reduce_default` seeds with the first product instead of 0, so `M`
    #products per element need only `M-1` additions; the announcer promotes
    #too, so the pinned 1s must land in `L`/`N` rather than be miscounted
    matmul(np.ones(sa), np.ones(sb), progress=('add', 'mul'))
    assert {(b.desc, b.total) for b in bars.instances} \
            == {('add', add), ('mul', mul)}

def test_matmul_draws_one_set_of_bars(bars):
    #one owner per call, whatever the operand ranks
    matmul(np.ones((2, 3)), np.ones((3, 4)), progress=('add', 'mul'))
    assert len(bars.instances) == 2



#scalar objects
def test_matmul_of_vectors_is_exact_for_fractions():
    v = np.array([Fraction(1, 3), Fraction(1, 6)], object)
    w = np.array([Fraction(3), Fraction(2)], object)
    assert matmul(v, w) == Fraction(4, 3)

def test_matmul_of_a_matrix_and_a_vector_is_exact_for_fractions():
    m = np.array([[Fraction(1, 2), Fraction(1, 3)]], object)
    v = np.array([Fraction(3), Fraction(3)], object)
    r = matmul(m, v)
    assert r.dtype == object and r[0] == Fraction(5, 2)

def test_outer_is_exact_for_fractions():
    v = np.array([Fraction(1, 3), Fraction(1, 6)], object)
    assert outer(v, v)[0, 0] == Fraction(1, 9)

def test_matmul_is_exact_for_fractions():
    m = np.array([[Fraction(1, 2), Fraction(1, 3)],
                  [Fraction(1, 4), Fraction(1, 5)]], object)
    r = matmul(m, m)
    assert r.dtype == object
    assert r[0, 0] == Fraction(1, 2)**2 + Fraction(1, 3)*Fraction(1, 4)



#promotion & edge cases
@pytest.mark.parametrize('v, w', [
    ([1, 2],                        [3, 4]),
    ([1, 2],                        [3.0, 4.0]),
    ([1+2j, 3],                     [1.0, 2.0]),
    (np.array([1, 2], np.int8),     np.array([3, 4], np.int8)),
])
def test_matmul_of_vectors_promotes_dtype_like_numpy(v, w):
    r = matmul(v, w)
    assert r == np.dot(v, w)
    assert np.result_type(r) == np.result_type(np.dot(v, w))

def test_empty_vector_product_is_zero():
    assert matmul([], []) == 0

def test_matmul_with_zero_inner_dimension():
    assert np.array_equal(matmul(np.ones((2, 0)), np.ones((0, 3))),
                          np.zeros((2, 3)))

def test_matmul_with_zero_inner_dimension_against_a_vector():
    assert np.array_equal(matmul(np.ones((2, 0)), np.ones(0)), np.zeros(2))



#errors
def test_outer_rejects_bad_shapes():
    with pytest.raises(ValueError):
        outer(np.zeros((2, 2)), np.zeros(2))

@pytest.mark.parametrize('v, w', [
    (np.zeros((2, 3)),    np.zeros((4, 2))),   #inner dimensions differ
    (np.zeros(2),         np.zeros(3)),        #vectors of different length
    (np.zeros((2, 3)),    np.zeros(4)),        #vector of the wrong length
    (np.zeros((2, 2, 2)), np.zeros((2, 2))),   #three dimensional
    (np.zeros(()),        np.zeros(2)),        #zero dimensional
])
def test_matmul_rejects_bad_shapes(v, w):
    with pytest.raises(ValueError):
        matmul(v, w)

@pytest.mark.parametrize('f, v, w', [
    (matmul, np.zeros(2),         np.zeros(3)),
    (matmul, np.zeros((2, 2, 2)), np.zeros((2, 2))),
    (outer,  np.zeros((2, 2)),    np.zeros(2)),
])
def test_errors_match_with_and_without_progress(bars, f, v, w):
    #the sanitiser & announcer must not raise a different error first
    with pytest.raises(ValueError) as bare:
        f(v, w)
    with pytest.raises(ValueError) as shown:
        f(v, w, progress=True)
    assert str(bare.value) == str(shown.value)
