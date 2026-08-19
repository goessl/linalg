from linalg.rank import *
from random import randint, binomialvariate
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

def _vrandq(shape:int|tuple[int,...]=1, grade:int=1000) -> npt.NDArray[object]:
    r = np.empty(shape, dtype=object)
    for i in np.ndindex(r.shape):
        r[i] = _binomq(grade)
    return r

def _vrandqr(M:int, N:int, R:int, grade:int=1000) -> npt.NDArray[object]:
    if not R: #an empty matmul would fill with int(0)
        return np.full((M, N), Fraction(0), dtype=object)
    return _vrandq((M, R), grade=grade) @ _vrandq((R, N), grade=grade)



@pytest.mark.filterwarnings('error')
def test_rank_decomp():
    for M in range(1, 10):
        for N in range(1, 10):
            for R in range(min(M, N)+1):
                A = _vrandqr(M, N, R)
                B, C = rank_decomp(A)
                assert B.shape==(A.shape[0], R) and C.shape==(R, A.shape[1])
                assert np.all(A == B@C)
                assert all(isinstance(B[i,j], Fraction)
                           for i, j in np.ndindex(B.shape))
                assert all(isinstance(C[i,j], Fraction)
                           for i, j in np.ndindex(C.shape))



#############################################################################
#appended by Claude - counting is always on, so any call here also checks
#its announcer. the `bars` fixture lives in conftest.py.
#############################################################################

from linalg.gauss import is_ref, ref_gauss_announce

#every category `Progress` knows, for asserting that the untracked ones
#are rejected rather than silently dropped
CATEGORIES = ('pos', 'neg', 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod')



def _holey(M, N, R, grade=5):
    """A rank `R` matrix with some columns zeroed.

    The holes matter: a plain rank `R` product is only deficient in its
    trailing columns, so `ref_gauss` only ever skips after the last pivot.
    A zeroed column in the middle makes it skip - and credit a pivot that
    never happens - while pivots are still being found.
    """
    A = _vrandqr(M, N, R, grade=grade) if R \
            else np.full((M, N), Fraction(0), dtype=object)
    for j in range(N):
        if randint(0, 3) == 0:
            A[:, j] = Fraction(0)
    return A



#the factorisation itself
def test_rank_decomp_columns_of_b_are_columns_of_a():
    #B is built by deleting non pivot columns, so every column it keeps must
    #still be the original, untouched column of A
    for _ in range(30):
        M, N = randint(1, 7), randint(1, 7)
        A = _holey(M, N, randint(0, min(M, N)))
        B, _ = rank_decomp(A)
        for j in range(B.shape[1]):
            assert any(np.all(B[:, j] == A[:, k]) for k in range(N))

def test_rank_decomp_c_is_in_reduced_row_echelon_form():
    #C is the nonzero rows of rref(A), so it is still reduced row echelon
    for _ in range(30):
        M, N = randint(1, 7), randint(1, 7)
        A = _holey(M, N, randint(0, min(M, N)))
        _, C = rank_decomp(A)
        assert is_ref(C, reduced=True)

def test_rank_decomp_c_has_no_zero_rows():
    for _ in range(30):
        M, N = randint(1, 7), randint(1, 7)
        _, C = rank_decomp(_holey(M, N, randint(0, min(M, N))))
        assert all(np.any(C[i, :]) for i in range(C.shape[0]))

def test_rank_decomp_reconstructs_holey_matrices_exactly():
    #test_rank_decomp only sees trailing rank deficiency, this also covers
    #the interior zero columns that drive the pivot crediting
    for _ in range(30):
        M, N = randint(1, 7), randint(1, 7)
        A = _holey(M, N, randint(0, min(M, N)))
        B, C = rank_decomp(A)
        assert np.all(A == B@C)



#in-place contract
def test_rank_decomp_does_not_consume_its_argument():
    #unlike `ref_gauss` it reduces a copy, which is why its signature is
    #written with `\to` and not `\mapsto`
    A = _vrandqr(4, 3, 2)
    before = A.copy()
    rank_decomp(A)
    assert np.all(A == before)

def test_rank_decomp_accepts_array_likes():
    B, C = rank_decomp([[1., 2.], [2., 4.]])
    assert np.allclose(B@C, [[1., 2.], [2., 4.]])



#announcements
@pytest.mark.parametrize('M, N, expected', [
    (1, 1, {('sub',  0), ('mul',  0), ('truediv',  1)}),
    (3, 3, {('sub', 18), ('mul', 18), ('truediv',  9)}),
    (4, 3, {('sub', 27), ('mul', 27), ('truediv',  9)}),
    (2, 5, {('sub', 10), ('mul', 10), ('truediv', 10)}),
    (5, 2, {('sub', 16), ('mul', 16), ('truediv',  4)}),
])
def test_rank_decomp_announces_the_reduced_ref_gauss_complexity(
        bars, M, N, expected):
    #Nr(M-1) sub, Nr(M-1) mul & Nr truediv for the largest possible rank
    #r = min(M, N) - `rank_decomp` adds no arithmetic of its own
    rank_decomp(np.random.rand(M, N), progress=True)
    assert {(b.desc, b.total) for b in bars.instances} == expected

def test_rank_decomp_announcement_is_the_reduced_ref_gauss_one(bars):
    #whichever announcer is wired, it must agree with `rank_decomp_announce`
    A = np.random.rand(5, 4)
    rank_decomp(A, progress=True)
    assert {(b.desc, b.total) for b in bars.instances} \
            == set(rank_decomp_announce(A).items()) \
            == set(ref_gauss_announce(A, reduced=True).items())

def test_rank_decomp_fills_its_bars_on_a_rank_deficient_matrix(bars):
    #the announcement assumes rank min(M,N); the pivots that turn out to be
    #impossible are credited by `ref_gauss` through the shared handler
    for _ in range(20):
        M, N = randint(1, 7), randint(1, 7)
        rank_decomp(_holey(M, N, randint(0, min(M, N))), progress=True)
    assert bars.instances and all(b.n == b.total for b in bars.instances)

def test_rank_decomp_untracked_category_raises():
    with pytest.raises(ValueError, match='untracked'):
        rank_decomp(np.random.rand(3, 3), progress=CATEGORIES)

def test_rank_decomp_bars_are_closed(bars):
    rank_decomp(np.random.rand(3, 3), progress=True)
    assert all(b.closed for b in bars.instances)

def test_rank_decomp_draws_nothing_by_default(bars):
    rank_decomp(np.random.rand(3, 3))
    assert bars.instances == []



#edge cases
@pytest.mark.parametrize('shape', [(0, 0), (0, 3), (3, 0)])
def test_rank_decomp_of_an_empty_matrix(shape):
    B, C = rank_decomp(np.empty(shape, dtype=object))
    assert B.shape == (shape[0], 0) and C.shape == (0, shape[1])

def test_rank_decomp_of_a_zero_matrix_has_rank_zero():
    A = np.full((3, 4), Fraction(0), dtype=object)
    B, C = rank_decomp(A)
    assert B.shape == (3, 0) and C.shape == (0, 4)



#sanitiser
def test_rank_decomp_sanitiser_returns_args_and_kwargs():
    args, kwargs = rank_decomp_sanitise([[1., 2.], [3., 4.]])
    A, = args
    assert isinstance(A, np.ndarray) and kwargs == {}

@pytest.mark.parametrize('A', [
    np.zeros(3),            #not two dimensional
    np.zeros((2, 2, 2)),    #not two dimensional
])
def test_rank_decomp_sanitiser_rejects_bad_shapes(A):
    with pytest.raises(ValueError):
        rank_decomp_sanitise(A)

def test_rank_decomp_sanitiser_is_idempotent():
    once, kwargs = rank_decomp_sanitise([[1., 2.], [3., 4.]])
    twice, kwargs_again = rank_decomp_sanitise(*once, **kwargs)
    assert np.array_equal(once[0], twice[0]) and kwargs == kwargs_again

@pytest.mark.parametrize('A', [np.zeros(3), np.zeros((2, 2, 2))])
def test_rank_decomp_errors_match_with_and_without_progress(A):
    #the sanitiser & announcer must not raise a different error first
    with pytest.raises(ValueError) as bare:
        rank_decomp(A)
    with pytest.raises(ValueError) as shown:
        rank_decomp(A, progress=True)
    assert str(bare.value) == str(shown.value)
