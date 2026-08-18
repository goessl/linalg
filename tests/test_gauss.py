from linalg.gauss import *
from random import binomialvariate
from fractions import Fraction
import numpy as np
import numpy.typing as npt
import pytest



def _binomz(sigma:int=1000) -> int:
    """Return a random binomial distributed integer with mean 0 and std sigma."""
    if sigma < 0:
        raise ValueError('sigma must be non-negative')
    return binomialvariate(4*sigma**2) - 2*sigma**2

def _binomq(grade:int=1000) -> Fraction:
    """Random centered binomial with variance one on a lattice of spacing grade."""
    if grade <= 0:
        raise ValueError('grade must be positive')
    return Fraction(_binomz(grade), grade)

def _vrandz(shape:int|tuple[int,...]=1, sigma:int=1000) -> npt.NDArray[object]:
    r = np.empty(shape, dtype=object)
    for i in np.ndindex(r.shape):
        r[i] = _binomz(sigma)
    return r

def _vrandq(shape:int|tuple[int,...]=1, grade:int=1000) -> npt.NDArray[object]:
    r = np.empty(shape, dtype=object)
    for i in np.ndindex(r.shape):
        r[i] = _binomq(grade)
    return r



@pytest.mark.filterwarnings('error')
def test_det_gauss():
    for N in range(20):
        A = np.random.rand(N, N)
        assert np.isclose(np.linalg.det(A), det_gauss(A))
    
    for N in range(20):
        A = _vrandq((N, N))
        actual = np.linalg.det(A.astype(float)) #actual before prediction
        prediction = det_gauss(A)               #because A gets mutated
        assert np.isclose(float(prediction), actual)
        #N=0 has no element to take the ring from, the empty product is `int`
        assert isinstance(prediction, Fraction) or N == 0

@pytest.mark.filterwarnings('error')
def test_inv_gauss():
    for N in range(20):
        A = np.random.rand(N, N)
        if not np.isclose(np.linalg.det(A), 0):
            assert np.allclose(np.linalg.inv(A), inv_gauss(A))
    
    for N in range(20):
        A = _vrandq((N, N))
        if det_gauss(A.copy()) != 0:
            assert np.allclose(np.linalg.inv(A.astype(float)),
                    inv_gauss(A).astype(float))
        else:
            with pytest.raises(ZeroDivisionError):
                inv_gauss(A)




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
@pytest.mark.parametrize('sanitiser', [det_gauss_sanitise, inv_gauss_sanitise])
def test_sanitiser_returns_args_and_kwargs(sanitiser):
    args, kwargs = sanitiser([[1., 2.], [3., 4.]])
    A, = args
    assert isinstance(A, np.ndarray) and kwargs == {}

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
    #it may run again on a nested call, so re-applying must be safe
    once, _ = sanitiser([[1., 2.], [3., 4.]])
    twice, _ = sanitiser(*once)
    assert np.array_equal(once[0], twice[0])



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
    A = _vrandq((5, 5))
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
