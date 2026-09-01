from linalg.triangular import *
from linalg.random import *
from fractions import Fraction
import numpy as np



def test_lu():
    tested = 0
    for M in range(1, 10):
        for N in range(1, 10):
            A = vrandq((M, N))
            try:
                L, U = lu(A.copy())
            except ZeroDivisionError:
                continue
            
            assert is_tril(L) and is_triu(U)
            assert np.array_equal(L@U, A)
            assert np.all(np.diag(L)==1) if M<=N else np.all(np.diag(U)==1)
            tested += 1
    assert tested > 20
    
    #https://en.wikipedia.org/wiki/LU_decomposition#Example
    L, U = lu(np.array([[Fraction(4), Fraction(3)],
                        [Fraction(6), Fraction(3)]]))
    assert np.array_equal(L, np.array([[Fraction(1),    Fraction(0)],
                                       [Fraction(3, 2), Fraction(1)]]))
    assert np.array_equal(U, np.array([[Fraction(4), Fraction(3)],
                                       [Fraction(0), Fraction(-3, 2)]]))

def test_plu():
    tested = 0
    for M in range(1, 10):
        for N in range(1, 10):
            A = vrandq((M, N))
            try:
                P, L, U = plu(A.copy())
            except ZeroDivisionError:
                continue
            
            assert is_perm(P) and is_tril(L) and is_triu(U)
            assert np.array_equal(P@L@U, A)
            assert np.all(np.diag(L) == 1)
            tested += 1
    assert tested > 20

def test_luq():
    tested = 0
    for M in range(1, 10):
        for N in range(1, 10):
            A = vrandq((M, N))
            try:
                L, U, Q = luq(A.copy())
            except ZeroDivisionError:
                continue
            
            assert is_tril(L) and is_triu(U) and is_perm(Q)
            assert np.array_equal(L@U@Q, A)
            assert np.all(np.diag(U) == 1)
            tested += 1
    assert tested > 20

def test_pluq():
    tested = 0
    for M in range(1, 10):
        for N in range(1, 10):
            A = vrandq((M, N))
            try:
                P, L, U, Q = pluq(A.copy())
            except ZeroDivisionError:
                continue
            
            assert is_perm(P) and is_tril(L) and is_triu(U) and is_perm(Q)
            assert np.array_equal(P@L@U@Q, A)
            assert np.all(np.diag(L)==1) if M<=N else np.all(np.diag(U)==1)
            tested += 1
    assert tested > 20



#############################################################################
#appended by Claude - counting is always on, so any call here also checks its
#announcer. the `bars` fixture lives in conftest.py.
#############################################################################

from random import randint
import pytest

#every category `Progress` knows, for asserting that the untracked ones
#are rejected rather than silently dropped
CATEGORIES = ('pos', 'neg', 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod')

DECOMPS = [
    pytest.param(lu,   lu_sanitise,   lu_announce,   id='lu'),
    pytest.param(plu,  plu_sanitise,  plu_announce,  id='plu'),
    pytest.param(luq,  luq_sanitise,  luq_announce,  id='luq'),
    pytest.param(pluq, pluq_sanitise, pluq_announce, id='pluq'),
]
EVERY = pytest.mark.parametrize('fn, sanitise, announce', DECOMPS)


def _product(factors):
    """Multiply the returned factors back into the original matrix.

    All four return them in product order, so one helper covers them all.
    """
    r = factors[0]
    for f in factors[1:]:
        r = r @ f
    return r



#predicates moved to test_util.py along with the functions themselves



#announcements
@pytest.mark.parametrize('fn, announce, shape, expected', [
    #lu & pluq both flip to the shorter side, so they announce alike
    (lu,   lu_announce,   (1, 1), ( 0,  0,  0)),
    (lu,   lu_announce,   (3, 3), ( 9,  9,  3)),
    (lu,   lu_announce,   (2, 5), ( 5,  5,  1)),
    (lu,   lu_announce,   (5, 2), ( 5,  5,  1)),
    (lu,   lu_announce,   (4, 7), (42, 42,  6)),
    (pluq, pluq_announce, (3, 3), ( 9,  9,  3)),
    (pluq, pluq_announce, (2, 5), ( 5,  5,  1)),
    (pluq, pluq_announce, (5, 2), ( 5,  5,  1)),
    (pluq, pluq_announce, (4, 7), (42, 42,  6)),
    #plu eliminates down columns and luq along rows, so their O differs and
    #the two disagree on every oblong shape
    (plu,  plu_announce,  (3, 3), ( 9,  9,  3)),
    (plu,  plu_announce,  (2, 5), ( 5,  5,  1)),
    (plu,  plu_announce,  (5, 2), (14, 14,  7)),
    (plu,  plu_announce,  (4, 7), (42, 42,  6)),
    (luq,  luq_announce,  (3, 3), ( 9,  9,  3)),
    (luq,  luq_announce,  (2, 5), (14, 14,  7)),
    (luq,  luq_announce,  (5, 2), ( 5,  5,  1)),
    (luq,  luq_announce,  (4, 7), (72, 72, 18)),
])
def test_announces_the_documented_complexity(bars, fn, announce, shape,
                                             expected):
    A = np.random.rand(*shape)
    sub, mul, truediv = expected
    assert announce(A) == {'sub': sub, 'mul': mul, 'truediv': truediv}
    fn(A, progress=True)
    assert {(b.desc, b.total) for b in bars.instances} \
            == {('sub', sub), ('mul', mul), ('truediv', truediv)}

def test_plu_and_luq_announce_each_others_transpose():
    #luq is plu run on the transpose, so the counts have to mirror
    for _ in range(20):
        A = np.random.rand(randint(1, 7), randint(1, 7))
        assert plu_announce(A) == luq_announce(A.T)

@EVERY
@pytest.mark.parametrize('shape', [(1, 1), (3, 3), (2, 5), (5, 2), (4, 7)])
def test_the_bars_end_exactly_full(bars, fn, sanitise, announce, shape):
    fn(np.random.rand(*shape), progress=True)
    assert bars.instances and all(b.n == b.total for b in bars.instances)

@pytest.mark.parametrize('shape', [(5, 2), (7, 3), (4, 1)])
def test_pluq_draws_one_set_of_bars_on_a_tall_matrix(bars, shape):
    #pluq recurses on the transpose when M>N; if that call does not get the
    #handler passed down it opens a second set of bars of its own and leaves
    #the outer ones stuck at zero
    pluq(np.random.rand(*shape), progress=True)
    assert [b.desc for b in bars.instances] == ['sub', 'mul', 'truediv']
    assert all(b.n == b.total for b in bars.instances)

@EVERY
def test_untracked_category_raises(fn, sanitise, announce):
    with pytest.raises(ValueError, match='untracked'):
        fn(np.random.rand(3, 3), progress=CATEGORIES)

@EVERY
def test_a_tracked_subset_is_accepted(bars, fn, sanitise, announce):
    fn(np.random.rand(4, 4), progress=('sub', 'truediv'))
    assert {b.desc for b in bars.instances} == {'sub', 'truediv'}

@EVERY
def test_the_bars_are_closed(bars, fn, sanitise, announce):
    fn(np.random.rand(3, 3), progress=True)
    assert all(b.closed for b in bars.instances)

@EVERY
def test_nothing_is_drawn_by_default(bars, fn, sanitise, announce):
    fn(np.random.rand(3, 3))
    assert bars.instances == []



#the transpose duality
def test_plu_and_luq_are_transposes_of_each_other():
    #both pick argmax|A[i:, i]|, one down the column and one along the row of
    #the transpose, so the same pivots get chosen and the factors mirror
    for M in range(1, 7):
        for N in range(1, 7):
            A = vrandq((M, N))
            P, L, U = plu(A.copy())
            L2, U2, Q2 = luq(A.T.copy())
            assert np.array_equal(Q2.T, P) and np.array_equal(U2.T, L) \
                    and np.array_equal(L2.T, U)

@pytest.mark.parametrize('fn', [lu, pluq], ids=['lu', 'pluq'])
def test_the_doolittle_crout_switch_is_a_transpose_duality(fn):
    #both flip to the shorter side, so transposing the input transposes the
    #answer - except on a square, where `M<=N` holds either way and the two
    #calls take the same Doolittle branch instead of mirroring
    for M in range(1, 7):
        for N in range(1, 7):
            if M == N:
                continue
            A = vrandq((M, N))
            forward = fn(A.copy())
            backward = fn(A.T.copy())
            assert all(np.array_equal(b, f.T)
                       for b, f in zip(backward, reversed(forward)))



#the singular path
@pytest.mark.parametrize('fn, A', [
    #lu does no pivoting at all, so a zero leading entry is already fatal
    pytest.param(lu, [[0., 1.], [1., 0.]], id='lu'),
    #plu searches down the column, so it takes a whole zero column
    pytest.param(plu, [[0., 1.], [0., 1.]], id='plu'),
    #luq searches along the row
    pytest.param(luq, [[0., 0.], [1., 1.]], id='luq'),
    #pluq searches the whole remaining block, so only a rank drop stops it
    pytest.param(pluq, [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]], id='pluq'),
])
def test_a_missing_pivot_raises(fn, A):
    with pytest.raises(ZeroDivisionError):
        fn(np.array(A))

@EVERY
def test_a_raise_still_closes_the_bars(bars, fn, sanitise, announce):
    #`visualisable` closes in a `finally` but checks only on the way out, so
    #a half filled run must close its bars without warning about the rest
    with pytest.raises(ZeroDivisionError):
        fn(np.zeros((4, 4)), progress=True)
    assert bars.instances and all(b.closed for b in bars.instances)



#the in-place contract
@EVERY
def test_an_ndarray_is_transformed_in_place(fn, sanitise, announce):
    A = np.array([[4., 3.], [6., 3.]])
    fn(A)
    assert not np.array_equal(A, [[4., 3.], [6., 3.]])

@EVERY
def test_an_array_like_leaves_the_original_alone(fn, sanitise, announce):
    #the sanitiser wraps a list into a fresh array, so there is nothing of
    #the caller's left to mutate - unlike `ref_gauss`, which may not wrap
    #because it answers through the mutation
    A = [[4., 3.], [6., 3.]]
    fn(A)
    assert A == [[4., 3.], [6., 3.]]



#edge cases
@EVERY
@pytest.mark.parametrize('shape', [(0, 0), (0, 3), (3, 0)])
def test_an_empty_matrix_still_factorises(fn, sanitise, announce, shape):
    assert _product(fn(np.empty(shape))).shape == shape

@EVERY
def test_a_one_by_one_matrix_needs_no_arithmetic(bars, fn, sanitise,
                                                 announce):
    fn(np.array([[7.]]), progress=True)
    assert all(b.total == 0 for b in bars.instances)



#exactness
@EVERY
def test_the_factors_reconstruct_fractions_exactly(fn, sanitise, announce):
    #no floating point anywhere - this is what the object dtype is for
    for _ in range(20):
        A = vrandq((randint(1, 6), randint(1, 6)))
        try:
            factors = fn(A.copy())
        except ZeroDivisionError:
            continue
        assert np.array_equal(_product(factors), A)



#sanitisers
@EVERY
def test_the_sanitiser_returns_args_and_kwargs(fn, sanitise, announce):
    args, kwargs = sanitise([[1., 2.], [3., 4.]])
    A, = args
    assert isinstance(A, np.ndarray) and kwargs == {}

@EVERY
def test_the_sanitiser_is_idempotent(fn, sanitise, announce):
    #`visualisable` re-runs it on a nested call, so it has to survive itself
    once, kwargs = sanitise([[1., 2.], [3., 4.]])
    twice, kwargs_again = sanitise(*once, **kwargs)
    assert np.array_equal(once[0], twice[0]) and kwargs == kwargs_again

@EVERY
@pytest.mark.parametrize('A', [np.zeros(3), np.zeros((2, 2, 2))])
def test_the_sanitiser_rejects_bad_dimensions(fn, sanitise, announce, A):
    with pytest.raises(ValueError):
        sanitise(A)

@EVERY
@pytest.mark.parametrize('A', [np.zeros(3), np.zeros((2, 2, 2))])
def test_the_errors_match_with_and_without_progress(fn, sanitise, announce,
                                                    A):
    #the sanitiser & announcer must not raise a different error first
    with pytest.raises(ValueError) as bare:
        fn(A)
    with pytest.raises(ValueError) as shown:
        fn(A, progress=True)
    assert str(bare.value) == str(shown.value)



#predicates
#back from test_util.py, following `is_perm`, `is_tril` & `is_triu` home
def test_is_perm_accepts_permutation_matrices():
    assert is_perm(np.eye(3, dtype=int))
    assert is_perm([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

@pytest.mark.parametrize('P', [
    [[2, 0], [0, 2]],   #ones scaled away
    [[1, 1], [0, 0]],   #column sums wrong
    [[1, 0], [1, 0]],   #row sums wrong
])
def test_is_perm_rejects_non_permutations(P):
    assert not is_perm(P)

def test_is_tril_and_is_triu():
    assert is_tril([[1, 0], [2, 3]]) and not is_triu([[1, 0], [2, 3]])
    assert is_triu([[1, 2], [0, 3]]) and not is_tril([[1, 2], [0, 3]])

def test_a_diagonal_matrix_is_both_triangular():
    #the predicates look strictly above/below, so the diagonal is shared
    assert is_tril(np.eye(4)) and is_triu(np.eye(4))

def test_the_predicates_take_the_non_square_overhang():
    #the factors are routinely oblong, so this is the shape they see
    assert is_tril([[1, 0, 0], [2, 3, 0]]) and is_triu([[1, 2, 3], [0, 4, 5]])

@pytest.mark.parametrize('predicate', [is_perm, is_tril, is_triu])
def test_the_predicates_accept_array_likes(predicate):
    predicate([[1, 0], [0, 1]])

@pytest.mark.parametrize('A', [np.zeros(3), np.zeros((2, 2, 2))])
@pytest.mark.parametrize('predicate', [is_perm, is_tril, is_triu])
def test_the_predicates_reject_bad_dimensions(predicate, A):
    with pytest.raises(ValueError):
        predicate(A)

def test_is_perm_rejects_a_non_square_matrix():
    #unlike the triangular pair, a permutation matrix has to be square
    with pytest.raises(ValueError):
        is_perm(np.zeros((2, 3)))

@pytest.mark.parametrize('predicate', [is_perm, is_tril, is_triu])
def test_an_empty_matrix_satisfies_every_predicate(predicate):
    #vacuously - and an empty matrix does come back out of all four
    assert predicate(np.empty((0, 0)))
