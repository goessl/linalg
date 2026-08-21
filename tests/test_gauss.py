from linalg.gauss import *
from linalg.random import *
from random import randint
from fractions import Fraction
import numpy as np
import pytest



@pytest.mark.filterwarnings('error')
def test_det_gauss():
    for N in range(10):
        A = np.random.rand(N, N)
        assert np.isclose(np.linalg.det(A), det_gauss(A))
    
    for N in range(10):
        for R in range(N+1):
            A = mrandqr(N, N, R)
            actual = np.linalg.det(A.astype(float)) #actual before prediction
            prediction = det_gauss(A, one=Fraction(1)) #because A gets mutated
            assert np.isclose(float(prediction), actual)
            assert isinstance(prediction, Fraction)

@pytest.mark.filterwarnings('error')
def test_inv_gauss():
    for N in range(10):
        A = np.random.rand(N, N)
        if not np.isclose(np.linalg.det(A), 0):
            assert np.allclose(np.linalg.inv(A), inv_gauss(A))
    
    for N in range(10):
        for R in range(N+1):
            A = mrandqr(N, N, R)
            if det_gauss(A.copy()) != 0:
                actual = np.linalg.inv(A.astype(float))
                prediction = inv_gauss(A)
                assert np.allclose(prediction.astype(float), actual)
            else:
                with pytest.raises(ZeroDivisionError):
                    inv_gauss(A)

@pytest.mark.filterwarnings('error')
@pytest.mark.parametrize('reduced', (False, True))
def test_ref_gauss(reduced):
    for M in range(1, 10):
        for N in range(1, 10):
            for R in range(min(M, N)+1):
                A = mrandqr(M, N, R)
                ref_gauss(A, reduced=reduced)
                assert is_ref(A, reduced=reduced)
    
    for _ in range(50):
        M, N = randint(1, 10), randint(1, 10)
        R = randint(0, min(M, N))
        A = mrandqr(M, N, R)
        rank = np.linalg.matrix_rank(A.astype(float))
        assert len(ref_gauss(A, reduced=reduced)) == rank




#############################################################################
#appended by Claude - counting is always on, so any call here also checks
#its announcer; these cover selection, exact totals & the sanitiser.
#the `bars` fixture lives in conftest.py.
#############################################################################

from linalg.leibniz import det_leibniz
from math import isclose

#every category `Progress` knows, for asserting that the untracked ones
#are rejected rather than silently dropped
CATEGORIES = ('pos', 'neg', 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod')

#complete pivoting needs an odd number of swaps here, so the determinant
#sign is flipped & `posneg` takes its `neg` branch
ANTIDIAG = np.array([[0., 2.],
                     [1., 0.]])



#swaps
def test_swap_rows():
    A = np.arange(6).reshape(3, 2)
    swap_rows(A, 0, 2)
    assert np.array_equal(A, [[4, 5], [2, 3], [0, 1]])

def test_swap_columns():
    A = np.arange(6).reshape(2, 3)
    swap_columns(A, 0, 2)
    assert np.array_equal(A, [[2, 1, 0], [5, 4, 3]])

def test_swap_pivot_swaps_both():
    A = np.arange(9).reshape(3, 3)
    swap_pivot(A, 0, 1, 2)
    #row 0 <-> row 1, then column 0 <-> column 2
    assert np.array_equal(A, [[5, 4, 3], [2, 1, 0], [8, 7, 6]])

@pytest.mark.parametrize('f', [swap_rows, swap_columns])
def test_swap_with_itself_changes_nothing(f):
    A = np.arange(6).reshape(3, 2)
    B = A.copy()
    f(A, 1, 1)
    assert np.array_equal(A, B)

@pytest.mark.parametrize('f', [swap_rows, swap_columns])
def test_swap_rejects_a_non_array(f):
    with pytest.raises(TypeError):
        f([[1, 2], [3, 4]], 0, 1)

def test_swap_pivot_rejects_a_non_array():
    with pytest.raises(TypeError):
        swap_pivot([[1, 2], [3, 4]], 0, 1, 1)

@pytest.mark.parametrize('f', [swap_rows, swap_columns])
def test_swap_rejects_bad_shapes(f):
    with pytest.raises(ValueError):
        f(np.zeros(3), 0, 1)

def test_swap_pivot_rejects_bad_shapes():
    with pytest.raises(ValueError):
        swap_pivot(np.zeros(3), 0, 1, 1)



#sanitiser
@pytest.mark.parametrize('sanitiser, extra', [
    (det_gauss_sanitise, ('one',)),
    (inv_gauss_sanitise, ()),
])
def test_sanitiser_returns_args_and_kwargs(sanitiser, extra):
    args, kwargs = sanitiser([[1., 2.], [3., 4.]])
    A, = args
    assert isinstance(A, np.ndarray) and tuple(kwargs) == extra

def test_det_gauss_sanitiser_resolves_the_ring():
    #`MISSING` is resolved here rather than in the executor, so the announcer
    #and the executor both receive a concrete element
    _, kwargs = det_gauss_sanitise([[1., 2.], [3., 4.]])
    assert kwargs['one'] == 1 and isinstance(kwargs['one'], np.float64)
    _, kwargs = det_gauss_sanitise(np.array([[Fraction(1)]], dtype=object))
    assert kwargs['one'] == 1 and isinstance(kwargs['one'], int)
    _, kwargs = det_gauss_sanitise([[1., 2.]] * 2, one=Fraction(1))
    assert kwargs['one'] == 1 and isinstance(kwargs['one'], Fraction)

def test_det_gauss_sanitiser_does_not_re_resolve():
    #it runs again on a nested call, so an already resolved `one` must survive
    once, kwargs = det_gauss_sanitise([[1., 2.], [3., 4.]], one=Fraction(1))
    _, again = det_gauss_sanitise(*once, **kwargs)
    assert again['one'] is kwargs['one']

@pytest.mark.parametrize('sanitiser', [det_gauss_sanitise, inv_gauss_sanitise])
@pytest.mark.parametrize('A', [
    np.zeros(3),            #not two dimensional
    np.zeros((2, 2, 2)),    #not two dimensional
    np.zeros((2, 3)),       #not square
])
def test_sanitiser_rejects_bad_shapes(sanitiser, A):
    with pytest.raises(ValueError):
        sanitiser(A)

@pytest.mark.parametrize('sanitiser', [det_gauss_sanitise, inv_gauss_sanitise])
def test_sanitiser_is_idempotent(sanitiser):
    #it may run again on a nested call, so feeding it its own output - kwargs
    #included, the way the decorator hands them on - must be safe
    once, kwargs = sanitiser([[1., 2.], [3., 4.]])
    twice, kwargs_again = sanitiser(*once, **kwargs)
    assert np.array_equal(once[0], twice[0]) and kwargs == kwargs_again



#announcements
@pytest.mark.parametrize('A', [
    np.eye(3),                      #no pivoting at all
    ANTIDIAG,                       #an odd number of swaps
    np.array([[0., 0., 3.],
              [0., 2., 0.],
              [1., 0., 0.]]),
    np.zeros((3, 3)),               #singular at the first pivot
    np.array([[1., 0.], [0., 0.]]), #singular at the second
])
def test_det_gauss_announces_the_awkward_pivotings(A):
    #counting is always on, so the call itself is the assertion - these are
    #the inputs a random matrix never produces: no pivoting at all, an odd
    #number of swaps, and both early exits, where `posneg` takes one branch
    #& the announced upper bound has to be topped up
    det_gauss(A.copy())

@pytest.mark.parametrize('N, expected', [
    (0, {('pos', 1), ('neg', 1), ('sub',  0), ('mul',  0), ('truediv', 0)}),
    (1, {('pos', 1), ('neg', 1), ('sub',  0), ('mul',  0), ('truediv', 0)}),
    (2, {('pos', 1), ('neg', 1), ('sub',  2), ('mul',  3), ('truediv', 1)}),
    (3, {('pos', 1), ('neg', 1), ('sub',  8), ('mul', 10), ('truediv', 3)}),
    (4, {('pos', 1), ('neg', 1), ('sub', 20), ('mul', 23), ('truediv', 6)}),
])
def test_det_gauss_announces_the_documented_complexity(bars, N, expected):
    #N(N^2-1)/3 sub, max(N(N^2+2)/3-1, 0) mul & N(N-1)/2 truediv
    det_gauss(np.random.rand(N, N), progress=True)
    assert {(b.desc, b.total) for b in bars.instances} == expected

@pytest.mark.parametrize('N, expected', [
    (1, {('sub',  0), ('mul',  0), ('truediv',  2)}),
    (2, {('sub',  8), ('mul',  8), ('truediv',  8)}),
    (3, {('sub', 36), ('mul', 36), ('truediv', 18)}),
    (4, {('sub', 96), ('mul', 96), ('truediv', 32)}),
])
def test_inv_gauss_announces_the_documented_complexity(bars, N, expected):
    #2N^3-2N^2 sub, 2N^3-2N^2 mul & 2N^2 truediv
    inv_gauss(np.random.rand(N, N), progress=True)
    assert {(b.desc, b.total) for b in bars.instances} == expected

def test_det_gauss_hands_its_handler_down(bars):
    #the inner vsub/outer/vtruediv count into the bars `det_gauss` owns
    det_gauss(np.random.rand(4, 4), progress=True)
    assert len(bars.instances) == 5

@pytest.mark.parametrize('f', [det_gauss, inv_gauss])
def test_untracked_category_raises(f):
    #neither tracks 'add', so naming it is a user mistake
    with pytest.raises(ValueError, match='untracked'):
        f(np.random.rand(3, 3), progress=CATEGORIES)

def test_det_gauss_tracked_subset_is_accepted(bars):
    det_gauss(np.random.rand(3, 3), progress=('sub', 'mul'))
    assert {b.desc for b in bars.instances} == {'sub', 'mul'}

@pytest.mark.parametrize('f', [det_gauss, inv_gauss])
def test_progress_false_draws_nothing(bars, f):
    f(np.random.rand(3, 3), progress=False)
    assert bars.instances == []

@pytest.mark.parametrize('f', [det_gauss, inv_gauss])
def test_naked_draws_nothing(bars, f):
    f(np.random.rand(3, 3))
    assert bars.instances == []

@pytest.mark.parametrize('f', [det_gauss, inv_gauss])
def test_bars_are_closed(bars, f):
    f(np.random.rand(3, 3), progress=True)
    assert all(b.closed for b in bars.instances)

def test_bars_are_closed_on_a_singular_inverse(bars):
    with pytest.raises(ZeroDivisionError):
        inv_gauss(np.zeros((3, 3)), progress=True)
    assert all(b.closed for b in bars.instances)



#scalar objects
def test_det_gauss_is_exact_for_fractions():
    A = np.array([[Fraction(1, 2), Fraction(1, 3)],
                  [Fraction(1, 4), Fraction(1, 5)]], object)
    r = det_gauss(A)
    assert r == Fraction(1, 2)*Fraction(1, 5) - Fraction(1, 3)*Fraction(1, 4)
    assert isinstance(r, Fraction)

def test_inv_gauss_is_exact_for_fractions():
    A = np.array([[Fraction(1, 2), Fraction(1, 3)],
                  [Fraction(1, 4), Fraction(1, 5)]], object)
    r = inv_gauss(A.copy())
    assert r.dtype == object
    #A^-1 A == I exactly, with no floating point anywhere
    assert np.array_equal(r @ A, np.eye(2, dtype=object))

def test_det_gauss_agrees_with_det_leibniz_exactly():
    #two unrelated algorithms over the same exact ring
    A = vrandq((5, 5))
    assert det_gauss(A.copy()) == det_leibniz(A.copy())



#edge cases
def test_empty_det_is_one():
    assert det_gauss(np.empty((0, 0))) == 1

def test_one_by_one_det_is_the_element():
    assert det_gauss([[7.]]) == 7

def test_empty_inverse_is_empty():
    assert inv_gauss(np.empty((0, 0))).shape == (0, 0)

def test_det_gauss_of_a_singular_matrix_is_zero():
    assert det_gauss(np.array([[1., 2.], [2., 4.]])) == 0

def test_det_gauss_sign_follows_the_pivoting():
    assert isclose(det_gauss(ANTIDIAG.copy()), -2)

def test_inv_gauss_undoes_column_pivoting():
    #the argsort at the end unpermutes `Q`, without it the rows of the
    #result would be left in pivot order
    assert np.allclose(inv_gauss(ANTIDIAG.copy()) @ ANTIDIAG, np.eye(2))



#in-place contract
@pytest.mark.parametrize('f', [det_gauss, inv_gauss])
def test_an_array_argument_is_consumed(f):
    #documented: the matrix is transformed in-place
    A = np.random.rand(3, 3)
    B = A.copy()
    f(A)
    assert not np.allclose(A, B)

@pytest.mark.parametrize('f', [det_gauss, inv_gauss])
def test_a_list_argument_is_not_consumed(f):
    #`asarray` copies a list, so only ndarray callers see the mutation
    L = [[4., 3.], [6., 3.]]
    f(L)
    assert L == [[4., 3.], [6., 3.]]



#errors
@pytest.mark.parametrize('f', [det_gauss, inv_gauss])
@pytest.mark.parametrize('A', [
    np.zeros(3),            #not two dimensional
    np.zeros((2, 2, 2)),    #not two dimensional
    np.zeros((2, 3)),       #not square
])
def test_rejects_bad_shapes(f, A):
    with pytest.raises(ValueError):
        f(A)

@pytest.mark.parametrize('f', [det_gauss, inv_gauss])
@pytest.mark.parametrize('A', [
    np.zeros(3),
    np.zeros((2, 2, 2)),
    np.zeros((2, 3)),
])
def test_errors_match_with_and_without_progress(bars, f, A):
    #the sanitiser & announcer must not raise a different error first
    with pytest.raises(ValueError) as bare:
        f(A)
    with pytest.raises(ValueError) as shown:
        f(A, progress=True)
    assert str(bare.value) == str(shown.value)

def test_singular_inverse_raises():
    with pytest.raises(ZeroDivisionError, match='singular'):
        inv_gauss(np.array([[1., 2.], [2., 4.]]))



#is_ref
#no library computes an unreduced row echelon form (it isn't unique), so the
#predicate is pinned by hand; the reduced half is cross checked below
@pytest.mark.parametrize('A, unreduced, reduced', [
    #                                          is_ref(A, False), is_ref(A, True)
    ([[]],                                     True,  True),   #no columns
    ([[0, 0], [0, 0]],                         True,  True),   #zero matrix
    ([[1, 0], [0, 1]],                         True,  True),   #identity
    ([[1, 5], [0, 0]],                         True,  True),   #free column right
    ([[1, 0, 3], [0, 1, 4]],                   True,  True),   #two pivots
    ([[0, 1, 0], [0, 0, 1]],                   True,  True),   #leading zero col
    ([[2, 0], [0, 1]],                         True,  False),  #pivot != 1
    ([[1, 0], [0, 2]],                         True,  False),  #pivot != 1
    ([[1, 1], [0, 1]],                         True,  False),  #nonzero above
    ([[1, 0], [1, 0]],                         False, False),  #pivots repeat
    ([[0, 1], [1, 0]],                         False, False),  #pivots descend
    ([[0, 0], [1, 0]],                         False, False),  #zero row first
    ([[1, 0], [0, 0], [0, 1]],                 False, False),  #zero row between
])
def test_is_ref(A, unreduced, reduced):
    A = np.array(A, dtype=object)
    assert is_ref(A, reduced=False) is unreduced
    assert is_ref(A, reduced=True) is reduced

def test_is_ref_accepts_array_likes():
    #unlike `ref_gauss` it only reads, so it may wrap with `asarray`
    assert is_ref([[1, 0], [0, 1]]) and is_ref(((1, 0), (0, 1)))

@pytest.mark.parametrize('A', [np.zeros(3), np.zeros((2, 2, 2))])
def test_is_ref_rejects_bad_shapes(A):
    with pytest.raises(ValueError):
        is_ref(A)



#ref_gauss against sympy
#the reduced form is unique and so are the pivot columns - they are a property
#of the matrix, not of the algorithm, which is why sympy pins the unreduced
#branch too even though its matrix is not unique
def _sympy(A):
    """Exact `sympy.Matrix` of a `Fraction`/`int` object array."""
    sympy = pytest.importorskip('sympy')
    return sympy.Matrix(A.shape[0], A.shape[1],
            [sympy.Rational(x.numerator, x.denominator) for x in A.flat])

def _holey(M, N, r, grade=5):
    """[`mrandqr`][linalg.random.mrandqr] with some columns zeroed.

    The holes matter: a plain rank `r` product is only deficient at the
    trailing columns, so every skip happens after the last pivot. A zeroed
    column in the middle makes `ref_gauss` skip and credit a lost pivot
    while pivots are still being found.
    """
    A = mrandqr(M, N, r, grade=grade)
    for j in range(N):
        if randint(0, 3) == 0:
            A[:, j] = Fraction(0)
    return A

@pytest.mark.parametrize('reduced', (False, True))
def test_ref_gauss_pivots_are_the_rref_pivots(reduced):
    for _ in range(50):
        M, N = randint(1, 6), randint(1, 6)
        A = _holey(M, N, randint(0, min(M, N)))
        expected = list(_sympy(A).rref()[1])
        assert ref_gauss(A, reduced=reduced) == expected

def test_ref_gauss_reduced_is_the_unique_rref():
    for _ in range(50):
        M, N = randint(1, 6), randint(1, 6)
        A = _holey(M, N, randint(0, min(M, N)))
        expected = _sympy(A).rref()[0]
        ref_gauss(A, reduced=True)
        assert _sympy(A) == expected



#ref_gauss announcements
@pytest.mark.parametrize('reduced', (False, True))
@pytest.mark.parametrize('shape', [(0, 0), (0, 3), (3, 0)])
def test_ref_gauss_of_an_empty_matrix(shape, reduced):
    assert ref_gauss(np.empty(shape, dtype=object), reduced=reduced) == []

@pytest.mark.parametrize('reduced', (False, True))
def test_ref_gauss_fills_its_bars_on_a_rank_deficient_matrix(bars, reduced):
    #the announcement assumes rank min(M,N); every pivot that turns out to be
    #impossible has to be credited, or the bars stop short of their total
    for _ in range(20):
        M, N = randint(1, 7), randint(1, 7)
        ref_gauss(_holey(M, N, randint(0, min(M, N))),
                  reduced=reduced, progress=True)
    assert bars.instances and all(b.n == b.total for b in bars.instances)

@pytest.mark.parametrize('M, N, expected', [
    (1, 1, {('sub',  0), ('mul',  0), ('truediv', 1)}),
    (3, 3, {('sub', 18), ('mul', 18), ('truediv', 9)}),
    (2, 5, {('sub', 10), ('mul', 10), ('truediv', 10)}),
    (5, 2, {('sub', 16), ('mul', 16), ('truediv',  4)}),
])
def test_ref_gauss_reduced_announces_the_documented_complexity(
        bars, M, N, expected):
    #Nr(M-1) sub, Nr(M-1) mul & Nr truediv for r = min(M, N)
    ref_gauss(np.random.rand(M, N), reduced=True, progress=True)
    assert {(b.desc, b.total) for b in bars.instances} == expected

@pytest.mark.parametrize('M, N, expected', [
    (1, 1, {('sub',  0), ('mul',  0), ('truediv', 0)}),
    (3, 3, {('sub',  9), ('mul',  9), ('truediv', 3)}),
    (2, 5, {('sub',  5), ('mul',  5), ('truediv', 1)}),
    (5, 2, {('sub', 14), ('mul', 14), ('truediv', 7)}),
])
def test_ref_gauss_unreduced_announces_the_documented_complexity(
        bars, M, N, expected):
    #Nr(2M-r-1)/2 sub, Nr(2M-r-1)/2 mul & r(2M-r-1)/2 truediv for r = min(M, N)
    ref_gauss(np.random.rand(M, N), reduced=False, progress=True)
    assert {(b.desc, b.total) for b in bars.instances} == expected

def test_ref_gauss_untracked_category_raises():
    with pytest.raises(ValueError, match='untracked'):
        ref_gauss(np.random.rand(3, 3), progress=CATEGORIES)

@pytest.mark.parametrize('reduced', (False, True))
def test_ref_gauss_bars_are_closed(bars, reduced):
    ref_gauss(np.random.rand(3, 3), reduced=reduced, progress=True)
    assert all(b.closed for b in bars.instances)



#ref_gauss sanitiser
def test_ref_gauss_sanitiser_returns_args_and_kwargs():
    A = np.array([[1., 2.], [3., 4.]])
    args, kwargs = ref_gauss_sanitise(A, False)
    assert args[0] is A and args[1] is False and kwargs == {}

def test_ref_gauss_sanitiser_keeps_the_array_identity():
    #`ref_gauss` writes through, so unlike the other sanitisers this one must
    #not `asarray` - a copy would silently drop the whole result
    A = np.array([[1., 2.], [3., 4.]])
    assert ref_gauss_sanitise(A)[0][0] is A

@pytest.mark.parametrize('A', [[[1., 2.]], ((1., 2.),), 5])
def test_ref_gauss_sanitiser_rejects_non_arrays(A):
    with pytest.raises(TypeError):
        ref_gauss_sanitise(A)

@pytest.mark.parametrize('A', [np.zeros(3), np.zeros((2, 2, 2))])
def test_ref_gauss_sanitiser_rejects_bad_shapes(A):
    with pytest.raises(ValueError):
        ref_gauss_sanitise(A)

def test_ref_gauss_errors_match_with_and_without_progress():
    with pytest.raises(ValueError) as bare:
        ref_gauss(np.zeros(3))
    with pytest.raises(ValueError) as shown:
        ref_gauss(np.zeros(3), progress=True)
    assert str(bare.value) == str(shown.value)
