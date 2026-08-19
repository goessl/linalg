from linalg.blas2 import *
from fractions import Fraction
import numpy as np
import pytest



@pytest.mark.filterwarnings('error')
def test_dot():
    for N in range(10):
        v, w = np.random.rand(N), np.random.rand(N)
        assert np.isclose(dot(v, w), v@w)

@pytest.mark.filterwarnings('error')
def test_dot_empty():
    #numpy types
    v, w = np.empty((0,), np.int64), np.empty((0,), np.float64)
    vw = dot(v, w)
    assert vw==0 and isinstance(vw, np.float64)
    
    #objects
    v, w = np.empty((0,), object), np.empty((0,), object)
    vw = dot(v, w)
    assert vw==0 and isinstance(vw, int)
    
    v, w = np.empty((0,), object), np.empty((0,), object)
    vw = dot(v, w, zero=Fraction())
    assert vw==0 and isinstance(vw, Fraction)

@pytest.mark.filterwarnings('error')
def test_outer():
    for M in range(10):
        for N in range(10):
            v, w = np.random.rand(M), np.random.rand(N)
            assert np.allclose(outer(v, w), np.outer(v, w))

@pytest.mark.filterwarnings('error')
def test_matmul():
    for L in range(10):
        for M in range(10):
            for N in range(10):
                v, w = np.random.rand(L, M), np.random.rand(M, N)
                assert np.allclose(matmul(v, w), v@w)

@pytest.mark.filterwarnings('error')
def test_matmul_empty():
    #L, M, N zero combinations already tested in test_matmul
    #only test empty sums and non empty result array here
    
    #numpy types
    A, B = np.empty((10, 0), np.int64), np.empty((0, 20), np.float64)
    AB = matmul(A, B)
    assert np.array_equal(AB, np.zeros((10, 20))) and AB.dtype==np.float64
    
    A, B = np.empty((0, 0), np.int64), np.empty((0, 0), np.float64)
    AB = matmul(A, B)
    assert np.array_equal(AB, np.zeros((0, 0))) and AB.dtype==np.float64
    
    #objects
    A, B = np.empty((10, 0), object), np.empty((0, 20), object)
    AB = matmul(A, B)
    assert all(AB[i,j]==0 and isinstance(AB[i,j], int)
               for i, j in np.ndindex(AB.shape))
    
    A, B = np.empty((10, 0), object), np.empty((0, 20), object)
    AB = matmul(A, B, zero=Fraction())
    assert all(AB[i,j]==0 and isinstance(AB[i,j], Fraction)
               for i, j in np.ndindex(AB.shape))



#############################################################################
#appended by Claude - counting is always on, so any call here also checks
#its announcer; these cover selection, exact totals & the sanitiser.
#the `bars` fixture lives in conftest.py.
#############################################################################

#every category `Progress` knows, for asserting that the untracked ones
#are rejected rather than silently dropped
CATEGORIES = ('pos', 'neg', 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod')



#announcements
@pytest.mark.parametrize('f, v, w', [
    (dot,    np.zeros(3),      np.zeros(3)),
    (outer,  np.zeros(3),      np.zeros(3)),
    (matmul, np.zeros((2, 2)), np.zeros((2, 2))),
])
def test_untracked_category_raises(f, v, w):
    #none of these track 'mod', so naming it is a user mistake
    with pytest.raises(ValueError, match='untracked'):
        f(v, w, progress=CATEGORIES)

def test_outer_does_not_track_add(bars):
    #`outer` announces 'mul' only, so 'add' must be rejected too
    with pytest.raises(ValueError, match='untracked'):
        outer(np.zeros(3), np.zeros(3), progress=('mul', 'add'))

def test_dot_needs_one_add_less_than_muls(bars):
    #`reduce_default` seeds with the first product instead of 0,
    #so `n` products need only `n-1` additions
    dot(np.arange(4.), np.arange(4.), progress=('add', 'mul'))
    assert {(b.desc, b.total) for b in bars.instances} == {('add', 3), ('mul', 4)}

def test_matmul_needs_one_add_less_per_element(bars):
    matmul(np.ones((2, 3)), np.ones((3, 4)), progress=('add', 'mul'))
    assert {(b.desc, b.total) for b in bars.instances} == {('add', 16), ('mul', 24)}

def test_matmul_hands_its_handler_to_dot(bars):
    #every inner `dot` counts into the two bars `matmul` owns
    matmul(np.ones((2, 3)), np.ones((3, 4)), progress=('add', 'mul'))
    assert len(bars.instances) == 2



#scalar objects
def test_dot_is_exact_for_fractions():
    v = np.array([Fraction(1, 3), Fraction(1, 6)], object)
    w = np.array([Fraction(3), Fraction(2)], object)
    assert dot(v, w) == Fraction(4, 3)

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
def test_dot_promotes_like_numpy(v, w):
    r = dot(v, w)
    assert r == np.dot(v, w)
    assert np.result_type(r) == np.result_type(np.dot(v, w))

def test_empty_dot_is_zero():
    assert dot([], []) == 0

def test_matmul_with_zero_inner_dimension():
    assert np.array_equal(matmul(np.ones((2, 0)), np.ones((0, 3))),
                          np.zeros((2, 3)))



#errors
@pytest.mark.parametrize('v, w', [
    (np.zeros((2, 2)), np.zeros((2, 2))),   #not one dimensional
    (np.zeros(2),      np.zeros(3)),        #different lengths
])
def test_dot_rejects_bad_shapes(v, w):
    with pytest.raises(ValueError):
        dot(v, w)

def test_outer_rejects_bad_shapes():
    with pytest.raises(ValueError):
        outer(np.zeros((2, 2)), np.zeros(2))

@pytest.mark.parametrize('v, w', [
    (np.zeros(3),      np.zeros(3)),        #not two dimensional
    (np.zeros((2, 3)), np.zeros((4, 2))),   #inner dimensions differ
])
def test_matmul_rejects_bad_shapes(v, w):
    with pytest.raises(ValueError):
        matmul(v, w)

@pytest.mark.parametrize('f, v, w', [
    (dot,    np.zeros(2),      np.zeros(3)),
    (matmul, np.zeros(3),      np.zeros(3)),
    (outer,  np.zeros((2, 2)), np.zeros(2)),
])
def test_errors_match_with_and_without_progress(bars, f, v, w):
    #the sanitiser & announcer must not raise a different error first
    with pytest.raises(ValueError) as bare:
        f(v, w)
    with pytest.raises(ValueError) as shown:
        f(v, w, progress=True)
    assert str(bare.value) == str(shown.value)
