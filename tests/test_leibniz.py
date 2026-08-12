from linalg.leibniz import *
import numpy as np



def test_det_leibniz():
    for N in range(8+1):
        A = np.random.rand(N, N)
        assert np.isclose(np.linalg.det(A), det_leibniz(A))



#############################################################################
#appended by Claude - the announce function only runs when `progress` is
#requested, so without these it is never executed by the suite at all.
#the `bars` fixture lives in conftest.py.
#############################################################################

from linalg.leibniz import permutations as lperms
from itertools import permutations as iperms
from math import factorial
from fractions import Fraction
import pytest
import warnings

OPS = ('pos', 'neg', 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod')


def sign(p):
    """Return the parity of `p` from its inversion count, the slow way."""
    return not sum(p[i] > p[j]
                   for i in range(len(p)) for j in range(i+1, len(p))) % 2



#permutations
@pytest.mark.parametrize('n', range(5+1))
def test_permutations_yields_itertools_permutations(n):
    assert [p for p, _ in lperms(range(n))] == list(iperms(range(n)))

@pytest.mark.parametrize('n', range(5+1))
def test_permutations_parity_matches_inversion_count(n):
    assert all(s == sign(p) for p, s in lperms(range(n)))

@pytest.mark.parametrize('n, r', [(n, r)
        for n in range(5+1) for r in range(n+1)])
def test_partial_permutations_yield_itertools_permutations(n, r):
    assert [p for p, _ in lperms(range(n), r)] == list(iperms(range(n), r))

def test_permutations_of_too_long_a_subsequence_is_empty():
    assert list(lperms(range(3), 4)) == []

def test_permutations_consumes_any_iterable():
    assert [p for p, _ in lperms(iter('abc'))] \
            == list(iperms('abc'))

def test_permutations_starts_even():
    #the identity comes first & is even, so `posneg` affirms it
    _, s = next(lperms(range(4)))
    assert s is True



#announcements match reality
@pytest.mark.parametrize('N', range(6+1))
def test_det_leibniz_announcement_matches_reality(bars, N):
    det_leibniz(np.random.rand(N, N), progress=OPS)
    assert bars.instances
    assert all(b.n == b.total for b in bars.instances)

@pytest.mark.parametrize('N', range(6+1))
def test_det_leibniz_warns_nothing(bars, N):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        det_leibniz(np.random.rand(N, N), progress=True)

@pytest.mark.parametrize('N, expected', [
    (0, {('pos', 1), ('neg',  0), ('add',  0), ('mul',  0)}),
    (1, {('pos', 1), ('neg',  0), ('add',  0), ('mul',  0)}),
    (2, {('pos', 1), ('neg',  1), ('add',  1), ('mul',  2)}),
    (3, {('pos', 3), ('neg',  3), ('add',  5), ('mul', 12)}),
    (4, {('pos',12), ('neg', 12), ('add', 23), ('mul', 72)}),
])
def test_det_leibniz_announces_the_documented_complexity(bars, N, expected):
    #ceil(n!/2) pos, floor(n!/2) neg, n!-1 add & (n-1)n! mul
    det_leibniz(np.random.rand(N, N), progress=True)
    assert {(b.desc, b.total) for b in bars.instances} == expected

def test_det_leibniz_signs_every_permutation(bars):
    #the affirmations & negations together cover all n! terms
    det_leibniz(np.random.rand(5, 5), progress=('pos', 'neg'))
    assert sum(b.total for b in bars.instances) == factorial(5)

def test_det_leibniz_needs_one_add_less_than_terms(bars):
    #`sum_default` seeds with the first term instead of 0,
    #so n! terms need only n!-1 additions
    det_leibniz(np.random.rand(4, 4), progress=('add',))
    bar, = bars.instances
    assert bar.total == factorial(4) - 1

def test_progress_true_draws_every_bar(bars):
    det_leibniz(np.random.rand(3, 3), progress=True)
    assert {b.desc for b in bars.instances} == {'pos', 'neg', 'add', 'mul'}

def test_progress_false_draws_nothing(bars):
    det_leibniz(np.random.rand(3, 3), progress=False)
    assert bars.instances == []

def test_naked_draws_nothing(bars):
    det_leibniz(np.random.rand(3, 3))
    assert bars.instances == []

def test_bars_are_closed(bars):
    det_leibniz(np.random.rand(3, 3), progress=True)
    assert all(b.closed for b in bars.instances)



#scalar objects
def test_det_leibniz_is_exact_for_fractions():
    A = np.array([[Fraction(1, 2), Fraction(1, 3)],
                  [Fraction(1, 4), Fraction(1, 5)]], object)
    r = det_leibniz(A)
    assert r == Fraction(1, 2)*Fraction(1, 5) - Fraction(1, 3)*Fraction(1, 4)
    assert isinstance(r, Fraction)



#edge cases
def test_empty_det_is_one():
    #the empty product over the single empty permutation
    assert det_leibniz(np.empty((0, 0))) == 1

def test_one_by_one_det_is_the_element():
    assert det_leibniz([[7]]) == 7

def test_det_leibniz_promotes_like_numpy():
    assert det_leibniz([[1, 2], [3, 4]]) == -2



#errors
@pytest.mark.parametrize('A', [
    np.zeros(3),            #not two dimensional
    np.zeros((2, 2, 2)),    #not two dimensional
    np.zeros((2, 3)),       #not square
])
def test_det_leibniz_rejects_bad_shapes(A):
    with pytest.raises(ValueError):
        det_leibniz(A)

@pytest.mark.parametrize('A', [
    np.zeros(3),
    np.zeros((2, 2, 2)),
    np.zeros((2, 3)),
])
def test_errors_match_with_and_without_progress(bars, A):
    #the announce function must not raise a different error first
    with pytest.raises(ValueError) as bare:
        det_leibniz(A)
    with pytest.raises(ValueError) as shown:
        det_leibniz(A, progress=True)
    assert str(bare.value) == str(shown.value)
