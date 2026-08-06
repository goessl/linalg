from linalg.blas2 import *
from random import randint
import numpy as np



def test_dot():
    for _ in range(10):
        N = randint(0, 10)
        v, w = np.random.rand(N), np.random.rand(N)
        assert np.isclose(dot(v, w), v@w)

def test_outer():
    for _ in range(10):
        M, N = randint(0, 10), randint(0, 10)
        v, w = np.random.rand(M), np.random.rand(N)
        assert np.allclose(outer(v, w), np.outer(v, w))

def test_matmul():
    for _ in range(10):
        L, M, N = randint(0, 10), randint(0, 10), randint(0, 10)
        v, w = np.random.rand(L, M), np.random.rand(M, N)
        assert np.allclose(matmul(v, w), v@w)



#############################################################################
#appended by Claude - the announce functions only run when `progress` is
#requested, so without these they are never executed by the suite at all.
#the `bars` fixture lives in conftest.py.
#############################################################################

from fractions import Fraction
import pytest

OPS = ('pos', 'neg', 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod')



#announcements match reality
def test_dot_announcement_matches_reality(bars):
    for _ in range(10):
        N = randint(0, 10)
        v, w = np.random.rand(N), np.random.rand(N)
        bars.instances = []
        dot(v, w, progress=OPS)
        assert all(b.n == b.total for b in bars.instances)

def test_outer_announcement_matches_reality(bars):
    for _ in range(10):
        M, N = randint(0, 10), randint(0, 10)
        v, w = np.random.rand(M), np.random.rand(N)
        bars.instances = []
        outer(v, w, progress=OPS)
        assert all(b.n == b.total for b in bars.instances)

def test_matmul_announcement_matches_reality(bars):
    for _ in range(10):
        L, M, N = randint(0, 10), randint(0, 10), randint(0, 10)
        v, w = np.random.rand(L, M), np.random.rand(M, N)
        bars.instances = []
        matmul(v, w, progress=OPS)
        assert all(b.n == b.total for b in bars.instances)

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
    #the announce function must not raise a different error first
    with pytest.raises(ValueError) as bare:
        f(v, w)
    with pytest.raises(ValueError) as shown:
        f(v, w, progress=OPS)
    assert str(bare.value) == str(shown.value)
