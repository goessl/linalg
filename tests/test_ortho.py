from linalg.ortho import *
from linalg.random import *
from random import randint
import pytest



def test_are_orthogonal_empty():
    assert are_orthogonal() == True
    assert are_orthogonal([]) == True
    assert are_orthogonal([], []) == True
    assert are_orthogonal([1]) == True
    assert are_orthogonal([1], [1]) == False

def test_is_normalised():
    assert is_normalised([]) == False
    assert is_normalised([1]) == True
    assert is_normalised([0]) == False
    assert is_normalised([2]) == False
    assert is_normalised([1, 0]) == True

@pytest.mark.filterwarnings('error')
def test_gram_schmidt():
    for _ in range(100):
        N = randint(0, 20)
        M = randint(0, N)
        vs = [vrandq(N) for _ in range(M)]
        gram_schmidt(vs)

        assert are_orthogonal(*vs)




#############################################################################
#appended by Claude - counting is always on, so any call here also checks
#its announcer; these cover the predicates, exact totals & the sanitiser.
#the `bars` fixture lives in conftest.py.
#############################################################################

from fractions import Fraction
import numpy as np

#every category `Progress` knows, for asserting that the untracked ones
#are rejected rather than silently dropped
CATEGORIES = ('pos', 'neg', 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod')



def _vs(M, N):
    """`M` almost surely independent float vectors of length `N`."""
    return [np.random.rand(N) for _ in range(M)]



#predicates
def test_are_orthogonal_checks_every_pair():
    #neighbours only would compare (e1, e2) & (e2, e1) and miss the repeat
    e1, e2 = [1, 0, 0], [0, 1, 0]
    assert are_orthogonal(e1, e2)
    assert not are_orthogonal(e1, e2, e1)

def test_zero_vectors_are_orthogonal_in_every_dimension():
    #the dot product is zero in each of these, and the empty sum is no
    #exception - dimension zero must not become a special case of its own
    assert are_orthogonal([0, 0], [0, 0])
    assert are_orthogonal([0, 0], [1, 1])
    assert are_orthogonal([], [])

def test_an_empty_vector_is_orthogonal_but_not_normalised():
    #`are_orthonormal` has to be False here, and normalisation is what decides
    #it: the empty vector has norm zero, orthogonality has nothing to object to
    assert are_orthogonal([], []) and not is_normalised([])
    assert not are_orthonormal([], [])

def test_orthogonality_does_not_imply_independence():
    #plain pairwise orthogonality admits the zero vector, so it is weaker than
    #the orthogonal basis `gram_schmidt` insists on - which is why the
    #`are_orthogonal` assertion in `test_gram_schmidt` is necessary but not
    #sufficient on its own
    assert are_orthogonal(np.zeros(2), np.zeros(2))
    with pytest.raises(ZeroDivisionError):
        gram_schmidt([np.zeros(2), np.zeros(2)])

@pytest.mark.parametrize('vs', [
    (),
    ([1, 0],),                  #normalised
    ([2, 0],),                  #not normalised
    ([1, 0], [0, 1]),           #orthonormal
    ([2, 0], [0, 1]),           #orthogonal, not normalised
    ([1, 0], [1, 0]),           #normalised, not orthogonal
    ([1, 0], [0, 1], [1, 0]),   #only the outer pair is bad
])
def test_are_orthonormal_agrees_with_its_parts(vs):
    #it repeats the `are_orthogonal` expression instead of calling it, so the
    #two can drift apart
    assert are_orthonormal(*vs) == (are_orthogonal(*vs)
            and all(is_normalised(v) for v in vs))

@pytest.mark.parametrize('f', [are_orthogonal, are_orthonormal])
def test_predicates_reject_mismatched_dimensionality(f):
    with pytest.raises(ValueError):
        f([1, 0], [1, 0, 0])

@pytest.mark.parametrize('f', [are_orthogonal, are_orthonormal, is_normalised])
def test_predicates_reject_non_vectors(f):
    with pytest.raises(ValueError):
        f(np.zeros((2, 2)))



#announcements
@pytest.mark.parametrize('M, N, expected', [
    (0, 0, {('add',  0), ('sub',  0), ('mul',  0), ('truediv', 0)}),
    (0, 3, {('add',  0), ('sub',  0), ('mul',  0), ('truediv', 0)}),
    (1, 1, {('add',  0), ('sub',  0), ('mul',  1), ('truediv', 0)}),
    (2, 2, {('add',  3), ('sub',  2), ('mul',  8), ('truediv', 1)}),
    (3, 3, {('add', 12), ('sub',  9), ('mul', 27), ('truediv', 3)}),
    (2, 5, {('add', 12), ('sub',  5), ('mul', 20), ('truediv', 1)}),
    (4, 4, {('add', 30), ('sub', 24), ('mul', 64), ('truediv', 6)}),
])
def test_gram_schmidt_announces_the_documented_complexity(
        bars, M, N, expected):
    #M(M+1)/2 max(N-1,0) add, M(M-1)N/2 sub, M^2 N mul & M(M-1)/2 truediv
    gram_schmidt(_vs(M, N), progress=True)
    assert {(b.desc, b.total) for b in bars.instances} == expected

def test_gram_schmidt_fills_its_bars_exactly(bars):
    #the announcement is exact rather than an upper bound, so nothing is ever
    #left to top up - the reason the executor needs no early exit credit
    for _ in range(20):
        N = randint(1, 7)
        gram_schmidt(_vs(randint(0, N), N), progress=True)
    assert bars.instances and all(b.n == b.total for b in bars.instances)

def test_gram_schmidt_does_not_credit_the_error_path(bars):
    #`_check` never runs when the executor raises, so unlike `det_gauss` -
    #whose early exit is a normal return - there is nothing to make up for,
    #and the bars are expected to stop short
    with pytest.raises(ZeroDivisionError):
        gram_schmidt([np.array([1., 0.]), np.array([2., 0.]),
                      np.array([0., 1.])], progress=True)
    assert any(b.n < b.total for b in bars.instances)
    assert all(b.closed for b in bars.instances)

def test_gram_schmidt_hands_its_handler_down(bars):
    #the inner matmul/vmul/vsub count into the bars `gram_schmidt` owns
    gram_schmidt(_vs(4, 4), progress=True)
    assert len(bars.instances) == 4

def test_gram_schmidt_untracked_category_raises():
    #it tracks no 'pos', 'neg', 'floordiv' or 'mod'
    with pytest.raises(ValueError, match='untracked'):
        gram_schmidt(_vs(3, 3), progress=CATEGORIES)

def test_gram_schmidt_tracked_subset_is_accepted(bars):
    gram_schmidt(_vs(3, 3), progress=('sub', 'mul'))
    assert {b.desc for b in bars.instances} == {'sub', 'mul'}

def test_gram_schmidt_progress_false_draws_nothing(bars):
    gram_schmidt(_vs(3, 3), progress=False)
    assert bars.instances == []

def test_gram_schmidt_naked_draws_nothing(bars):
    gram_schmidt(_vs(3, 3))
    assert bars.instances == []

def test_gram_schmidt_bars_are_closed(bars):
    gram_schmidt(_vs(3, 3), progress=True)
    assert all(b.closed for b in bars.instances)



#sanitiser
def test_gram_schmidt_sanitiser_returns_args_and_kwargs():
    args, kwargs = gram_schmidt_sanitise([[1., 0.], [1., 1.]])
    vs, = args
    assert all(isinstance(v, np.ndarray) for v in vs) and kwargs == {}

def test_gram_schmidt_sanitiser_keeps_the_list_identity():
    #it fills the caller's list rather than building a new one, which is what
    #carries the result back out for a list of lists
    vs = [[1., 0.], [1., 1.]]
    assert gram_schmidt_sanitise(vs)[0][0] is vs

def test_gram_schmidt_sanitiser_keeps_the_array_identity():
    #`gram_schmidt` writes through the entries, so an ndarray has to survive
    #`asarray` untouched - a copy would silently drop the whole result
    v = np.array([1., 0.])
    assert gram_schmidt_sanitise([v])[0][0][0] is v

def test_gram_schmidt_sanitiser_is_idempotent():
    #it may run again on a nested call, so feeding it its own output - kwargs
    #included, the way the decorator hands them on - must be safe
    once, kwargs = gram_schmidt_sanitise([[1., 0.], [1., 1.]])
    twice, kwargs_again = gram_schmidt_sanitise(*once, **kwargs)
    assert twice[0] is once[0] and kwargs == kwargs_again

@pytest.mark.parametrize('vs', [
    [np.zeros((2, 2))],             #not one dimensional
    [np.zeros(2), np.zeros(3)],     #ragged
])
def test_gram_schmidt_sanitiser_rejects_bad_shapes(vs):
    with pytest.raises(ValueError):
        gram_schmidt_sanitise(vs)

def test_gram_schmidt_sanitiser_rejects_immutable_sequences():
    #it fills the sequence in place, so a tuple never gets as far as the
    #shape checks
    with pytest.raises(TypeError):
        gram_schmidt_sanitise((np.zeros(2),))



#in-place contract
def test_gram_schmidt_writes_through_its_arrays():
    #documented: the transformation happens in-place
    a, b = np.array([1., 0.]), np.array([1., 1.])
    gram_schmidt([a, b])
    assert np.allclose(a, [1., 0.]) and np.allclose(b, [0., 1.])

def test_gram_schmidt_fills_a_list_of_lists():
    #`asarray` copies a list, but the sanitiser writes the copy back into the
    #caller's list, so unlike `det_gauss` a list argument is consumed too
    vs = [[1., 0.], [1., 1.]]
    gram_schmidt(vs)
    assert all(isinstance(v, np.ndarray) for v in vs)
    assert np.allclose(vs[1], [0., 1.])



#scalar objects
def test_gram_schmidt_is_exact_for_fractions():
    vs = [np.array([Fraction(1), Fraction(1)], object),
          np.array([Fraction(1), Fraction(0)], object)]
    gram_schmidt(vs)
    #[1, 0] - 1/2 [1, 1], with no floating point anywhere
    assert vs[1].dtype == object
    assert list(vs[1]) == [Fraction(1, 2), Fraction(-1, 2)]
    assert are_orthogonal(*vs)

def test_gram_schmidt_orthogonalises_without_normalising():
    #the name says orthogonalise; the lengths are left alone
    vs = [np.array([3., 0.]), np.array([1., 1.])]
    gram_schmidt(vs)
    assert are_orthogonal(*vs) and not are_orthonormal(*vs)



#edge cases
def test_gram_schmidt_of_no_vectors_does_nothing():
    vs = []
    gram_schmidt(vs)
    assert vs == []

def test_gram_schmidt_of_a_single_vector_leaves_it_alone():
    v = np.array([3., 4.])
    gram_schmidt([v])
    assert np.allclose(v, [3., 4.])

def test_gram_schmidt_of_empty_vectors_raises():
    #a zero length vector has a zero norm, so it can never span anything
    with pytest.raises(ZeroDivisionError):
        gram_schmidt([np.empty(0)])



#errors
def test_gram_schmidt_dependent_vectors_raise():
    with pytest.raises(ZeroDivisionError, match='orthogonalisable'):
        gram_schmidt([np.array([1., 1.]), np.array([2., 2.])])

@pytest.mark.parametrize('vs', [
    [np.zeros((2, 2))],
    [np.zeros(2), np.zeros(3)],
])
def test_gram_schmidt_errors_match_with_and_without_progress(bars, vs):
    #the sanitiser & announcer must not raise a different error first
    with pytest.raises(ValueError) as bare:
        gram_schmidt(list(vs))
    with pytest.raises(ValueError) as shown:
        gram_schmidt(list(vs), progress=True)
    assert str(bare.value) == str(shown.value)
