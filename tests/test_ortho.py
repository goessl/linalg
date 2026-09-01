from linalg.ortho import *
from linalg.random import *
from linalg.triangular import is_triu
from random import randint
import numpy as np
import sympy as sp
import pytest



def test_are_orthogonal_empty():
    assert are_orthogonal([[]]) == True
    assert are_orthogonal([[],
                           []]) == True
    assert are_orthogonal([[1]]) == True
    assert are_orthogonal([[1],
                           [1]]) == False

def test_are_normalised():
    assert are_normalised([]) == False
    assert are_normalised([1]) == True
    assert are_normalised([0]) == False
    assert are_normalised([2]) == False
    assert are_normalised([1, 0]) == True


def test_is_orthogonal():
    #https://en.wikipedia.org/wiki/Orthogonal_matrix#Examples
    assert is_orthogonal([[1, 0],
                          [0, 1]]) == True
    assert is_orthogonal([[1, 0],
                          [0, -1]]) == True
    assert is_orthogonal([[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0]]) == True

def test_is_semiorthogonal():
    #https://en.wikipedia.org/wiki/Orthogonal_matrix#Examples
    assert is_semiorthogonal([[1, 0],
                              [0, 1]]) == True
    assert is_semiorthogonal([[1, 0],
                              [0, -1]]) == True
    assert is_semiorthogonal([[0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [1, 0, 0, 0],
                              [0, 1, 0, 0]]) == True
    #https://en.wikipedia.org/wiki/Semi-orthogonal_matrix#Examples
    assert is_semiorthogonal([[1, 0],
                              [0, 1],
                              [0, 0]]) == True
    assert is_semiorthogonal([[1, 0, 0],
                              [0, 1, 0]]) == True
    assert is_semiorthogonal([[2, 0],
                              [0, 1],
                              [0, 0]]) == True

def test_is_orthonormal():
    #https://en.wikipedia.org/wiki/Orthogonal_matrix#Examples
    assert is_orthonormal([[1, 0],
                           [0, 1]]) == True
    assert is_orthonormal([[1, 0],
                           [0, -1]]) == True
    assert is_orthonormal([[0, 0, 0, 1],
                           [0, 0, 1, 0],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0]]) == True

def test_is_semiorthonormal():
    #https://en.wikipedia.org/wiki/Orthogonal_matrix#Examples
    assert is_semiorthonormal([[1, 0],
                               [0, 1]]) == True
    assert is_semiorthonormal([[1, 0],
                               [0, -1]]) == True
    assert is_semiorthonormal([[0, 0, 0, 1],
                               [0, 0, 1, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0]]) == True
    #https://en.wikipedia.org/wiki/Semi-orthogonal_matrix#Examples
    assert is_semiorthonormal([[1, 0],
                               [0, 1],
                               [0, 0]]) == True
    assert is_semiorthonormal([[1, 0, 0],
                               [0, 1, 0]]) == True
    assert is_semiorthonormal([[2, 0],
                               [0, 1],
                               [0, 0]]) == False


@pytest.mark.filterwarnings('error')
def test_gram_schmidt():
    for _ in range(100):
        N = randint(0, 20)
        M = randint(0, N)
        vs = vrandq((M, N))
        gram_schmidt(vs)
        
        assert are_orthogonal(vs)
    
    for _ in range(10):
        N = randint(0, 10)
        M = randint(0, N)
        vs = np.vectorize(sp.sympify, 'O')(vrandq((M, N)))
        gram_schmidt(vs, sqrt=sp.sqrt)
        
        assert are_orthonormal(vs)


@pytest.mark.filterwarnings('error')
def test_qr_decomp():
    for _ in range(100):
        N = randint(0, 20)
        A = vrandq((N, N))
        Q, R = qr_decomp(A.copy())
        
        assert np.array_equal(Q@R, A) and is_orthogonal(Q) and is_triu(R)
    
    for _ in range(10):
        N = randint(0, 10)
        A = np.vectorize(sp.sympify, 'O')(vrandq((N, N)))
        Q, R = qr_decomp(A.copy(), sqrt=sp.sqrt)
        
        assert np.array_equal(Q@R, A) and is_orthonormal(Q) and is_triu(R)



#############################################################################
#appended by Claude - counting is always on, so any call here also checks
#its announcer; these cover the predicates, exact totals & the sanitiser.
#the `bars` fixture lives in conftest.py.
#############################################################################

from fractions import Fraction
import math

#every category `Progress` knows, for asserting that the untracked ones
#are rejected rather than silently dropped
CATEGORIES = ('pos', 'neg', 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod')



def _vs(M, N):
    """`M` almost surely independent float row vectors of length `N`."""
    return np.random.rand(M, N)

def _exact_sqrt(x):
    """Square root of a `Fraction` with a square numerator & denominator."""
    return Fraction(math.isqrt(x.numerator), math.isqrt(x.denominator))



#predicates
def test_are_orthogonal_checks_every_pair():
    #neighbours only would compare (e1, e2) & (e2, e1) and miss the repeat
    assert are_orthogonal([[1, 0, 0], [0, 1, 0]])
    assert not are_orthogonal([[1, 0, 0], [0, 1, 0], [1, 0, 0]])

def test_zero_vectors_are_orthogonal_in_every_dimension():
    #the dot product is zero in each of these, and the empty sum is no
    #exception - dimension zero must not become a special case of its own
    assert are_orthogonal([[0, 0], [0, 0]])
    assert are_orthogonal([[0, 0], [1, 1]])
    assert are_orthogonal([[], []])

def test_an_empty_vector_is_orthogonal_but_not_normalised():
    #`are_orthonormal` has to be False here, and normalisation is what decides
    #it: the empty vector has norm zero, orthogonality has nothing to object to
    assert are_orthogonal([[], []]) and not are_normalised([])
    assert not are_orthonormal([[], []])

def test_orthogonality_does_not_imply_independence():
    #plain pairwise orthogonality admits the zero vector, so it is weaker than
    #the orthogonal basis `gram_schmidt` insists on - which is why the
    #`are_orthogonal` assertion in `test_gram_schmidt` is necessary but not
    #sufficient on its own
    assert are_orthogonal(np.zeros((2, 2)))
    with pytest.raises(ZeroDivisionError):
        gram_schmidt(np.zeros((2, 2)))

@pytest.mark.parametrize('vs, expected', [
    (np.empty((0, 2)), True),           #nothing to violate either half
    ([[1, 0]], True),                   #normalised
    ([[2, 0]], False),                  #not normalised
    ([[1, 0], [0, 1]], True),           #orthonormal
    ([[2, 0], [0, 1]], False),          #orthogonal, not normalised
    ([[1, 0], [1, 0]], False),          #normalised, not orthogonal
    ([[1, 0], [0, 1], [1, 0]], False),  #only the outer pair is bad
])
def test_are_orthonormal_needs_both_halves(vs, expected):
    #it is a conjunction of two predicates, so neither half may go missing
    assert are_orthonormal(vs) == expected

@pytest.mark.parametrize('f', [are_orthogonal, are_orthonormal])
def test_predicates_reject_a_loose_vector(f):
    #the vectors arrive as the rows of one array now, so a bare vector is a
    #shape error rather than a call with a single vector
    with pytest.raises(ValueError, match='two dimensional'):
        f([1, 0])

@pytest.mark.parametrize('f', [are_orthogonal, are_orthonormal])
def test_predicates_reject_ragged_rows(f):
    with pytest.raises(ValueError):
        f([[1, 0], [1, 0, 0]])

def test_are_normalised_takes_one_vector_or_many():
    #the one predicate that dispatches on the dimension instead of demanding
    #rows, so both arities have to keep working
    assert are_normalised([1, 0])
    assert are_normalised([[1, 0], [0, 1]])
    assert not are_normalised([[1, 0], [2, 0]])
    with pytest.raises(ValueError, match='one or two dimensional'):
        are_normalised(np.zeros((2, 2, 2)))

def test_are_normalised_separates_an_empty_vector_from_no_vectors():
    #the dimension decides: a zero length vector has norm zero, while zero
    #vectors leave nothing to violate
    assert not are_normalised([])
    assert are_normalised(np.empty((0, 2)))


#matrix predicates
@pytest.mark.parametrize('f', [is_orthogonal, is_orthonormal])
def test_the_square_predicates_reject_rectangles(f):
    #the semi variants are the ones that take any shape
    with pytest.raises(ValueError, match='square'):
        f([[1, 0], [0, 1], [0, 0]])

def test_is_orthogonal_needs_both_sides():
    #without normalisation orthogonal rows do not imply orthogonal columns,
    #which is why the conjunction cannot be halved the way `is_orthonormal` is
    Q = [[1, 1], [2, -2]]
    assert are_orthogonal(Q) and not are_orthogonal(np.transpose(Q))
    assert not is_orthogonal(Q)
    assert is_semiorthogonal(Q)

def test_is_orthonormal_only_tests_the_rows_but_the_columns_follow():
    #for a square matrix Q^T Q = I forces Q Q^T = I, so the second sweep the
    #other predicates need is redundant here
    Q = np.array([[Fraction(3, 5), Fraction(4, 5)],
                  [Fraction(-4, 5), Fraction(3, 5)]], object)
    assert is_orthonormal(Q) and are_orthonormal(Q.T)

def test_orthogonal_does_not_mean_orthonormal():
    #this module drops the normalisation that the cited definition of an
    #orthogonal matrix includes
    assert is_orthogonal([[1, 0], [0, 2]])
    assert not is_orthonormal([[1, 0], [0, 2]])

def test_is_semiorthonormal_does_not_mix_the_sides():
    #orthogonal rows together with normalised columns is neither side being
    #orthonormal, so the two halves must be tested per side
    Q = [[1, 1], [0, 0]]
    assert are_orthogonal(Q) and are_normalised(np.transpose(Q))
    assert not is_semiorthonormal(Q)

@pytest.mark.parametrize('f', [is_semiorthogonal, is_semiorthonormal])
@pytest.mark.parametrize('Q', [
    [[1, 0], [0, 1], [0, 0]],
    [[2, 0], [0, 1], [0, 0]],
    [[1, 1], [2, -2]],
    [[1, 1], [0, 0]]
])
def test_the_semi_predicates_are_transpose_symmetric(f, Q):
    #they ask about either side, so which one is the longer must not matter
    assert f(Q) == f(np.transpose(Q))



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

def test_gram_schmidt_announcer_reads_the_shape_in_order():
    #`gram_schmidt_cost` takes (vectors, dimension), which is the shape of the
    #array and not its transpose - these two disagree, so the order is pinned
    assert gram_schmidt_cost(2, 5) != gram_schmidt_cost(5, 2)
    assert gram_schmidt_announce(np.zeros((2, 5))) == gram_schmidt_cost(2, 5)

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
        gram_schmidt(np.array([[1., 0.], [2., 0.], [0., 1.]]), progress=True)
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
    args, kwargs = gram_schmidt_sanitise(np.array([[1., 0.], [1., 1.]]))
    vs, = args
    assert isinstance(vs, np.ndarray) and vs.ndim == 2
    assert kwargs == {'sqrt': None}

def test_gram_schmidt_sanitiser_passes_the_sqrt_on():
    #the announcer picks the cost by it, so it has to survive the sanitiser
    _, kwargs = gram_schmidt_sanitise(np.array([[1., 0.], [1., 1.]]),
                                      sqrt=math.sqrt)
    assert kwargs == {'sqrt': math.sqrt}

def test_gram_schmidt_sanitiser_keeps_the_array_identity():
    #`gram_schmidt` writes through the rows, so an ndarray has to survive
    #`asarray` untouched - a copy would silently drop the whole result
    vs = np.array([[1., 0.], [1., 1.]])
    assert gram_schmidt_sanitise(vs)[0][0] is vs

def test_gram_schmidt_sanitiser_is_idempotent():
    #it may run again on a nested call, so feeding it its own output - kwargs
    #included, the way the decorator hands them on - must be safe
    once, kwargs = gram_schmidt_sanitise(np.array([[1., 0.], [1., 1.]]))
    twice, kwargs_again = gram_schmidt_sanitise(*once, **kwargs)
    assert twice[0] is once[0] and kwargs == kwargs_again

@pytest.mark.parametrize('vs', [
    [[1., 0.], [1., 1.]],   #a nested list
    ((1., 0.), (1., 1.)),   #a nested tuple
])
def test_gram_schmidt_sanitiser_rejects_what_it_cannot_write_through(vs):
    #`asarray` would copy these and the orthogonalisation would be lost, so
    #they are refused outright rather than silently thrown away - the guard
    #`ref_gauss_sanitise` and `swap_rows` use for the same reason
    with pytest.raises(TypeError, match='numpy.ndarray'):
        gram_schmidt_sanitise(vs)

@pytest.mark.parametrize('vs', [
    np.zeros(2),            #a bare vector
    np.zeros((2, 2, 2)),    #a stack of matrices
])
def test_gram_schmidt_sanitiser_rejects_bad_shapes(vs):
    with pytest.raises(ValueError, match='two dimensional'):
        gram_schmidt_sanitise(vs)

def test_gram_schmidt_sanitiser_rejects_ragged_rows():
    #ragged rows only survive `np.array` as an object array of rows, which is
    #one dimensional, so the shape check is what turns them away
    with pytest.raises(ValueError, match='two dimensional'):
        gram_schmidt_sanitise(np.array([[1., 0.], [1., 0., 0.]], dtype=object))



#in-place contract
def test_gram_schmidt_writes_through_its_array():
    #documented: the transformation happens in-place
    vs = np.array([[1., 0.], [1., 1.]])
    gram_schmidt(vs)
    assert np.allclose(vs, [[1., 0.], [0., 1.]])

def test_gram_schmidt_rejects_a_list_rather_than_losing_the_result():
    #a copy could only ever be orthogonalised and dropped, so the in-place
    #contract is enforced instead of merely documented - `ref_gauss` is the
    #same, and neither can silently do nothing
    with pytest.raises(TypeError, match='numpy.ndarray'):
        gram_schmidt([[1., 0.], [1., 1.]])



#return value
def test_gram_schmidt_returns_the_squared_norms():
    vs = np.array([[3., 0.], [1., 1.]])
    dots = gram_schmidt(vs)
    #[1, 1] - 1/3 [3, 0] = [0, 1]
    assert np.allclose(vs, [[3., 0.], [0., 1.]])
    assert np.allclose(dots, [9., 1.])

def test_gram_schmidt_returns_one_norm_per_vector():
    for M in range(4):
        assert len(gram_schmidt(_vs(M, 4))) == M

def test_gram_schmidt_norms_describe_the_output_not_the_input():
    #the denominators handed back are the ones it divided by, so they belong
    #to the orthogonalised rows
    vs = _vs(4, 5)
    dots = gram_schmidt(vs)
    assert np.allclose(dots, [v@v for v in vs])



#scalar objects
def test_gram_schmidt_is_exact_for_fractions():
    vs = np.array([[Fraction(1), Fraction(1)],
                   [Fraction(1), Fraction(0)]], object)
    dots = gram_schmidt(vs)
    #[1, 0] - 1/2 [1, 1], with no floating point anywhere
    assert vs.dtype == object
    assert list(vs[1]) == [Fraction(1, 2), Fraction(-1, 2)]
    assert dots == [Fraction(2), Fraction(1, 2)]
    assert are_orthogonal(vs)

def test_gram_schmidt_orthogonalises_without_normalising():
    #the name says orthogonalise; the lengths are left alone
    vs = np.array([[3., 0.], [1., 1.]])
    gram_schmidt(vs)
    assert are_orthogonal(vs) and not are_orthonormal(vs)



#sqrt
def test_gram_schmidt_with_sqrt_orthonormalises_exactly():
    #both norms are rational here, so the whole normalisation stays in Q and
    #`are_orthonormal` can be asserted exactly rather than approximately
    vs = np.array([[Fraction(3), Fraction(4)],
                   [Fraction(1), Fraction(0)]], object)
    #[3, 4]/5, then [1, 0] - 3/5 [3, 4]/5 = [16, -12]/25 of length 4/5
    dots = gram_schmidt(vs, sqrt=_exact_sqrt)
    assert are_orthonormal(vs)
    assert list(vs[1]) == [Fraction(4, 5), Fraction(-3, 5)]
    assert dots == [Fraction(25), Fraction(16, 25)]

def test_gram_schmidt_with_sqrt_returns_the_squares_it_divided_by():
    #the norms belong to the orthogonalised rows before normalisation, so both
    #paths hand back the same squares and end up at the same vectors
    vs = np.array([[3., 0.], [1., 1.]])
    dots = gram_schmidt(vs, sqrt=math.sqrt)
    assert np.allclose(dots, [9., 1.])
    plain = np.array([[3., 0.], [1., 1.]])
    for v, d in zip(plain, gram_schmidt(plain)):
        v /= math.sqrt(d)
    assert np.allclose(plain, vs)

def test_gram_schmidt_with_sqrt_still_rejects_dependent_vectors():
    #the zero check guards the normalising division too, so it must not fall
    #through to whatever the caller's sqrt does with a zero norm
    vs = np.array([[Fraction(3), Fraction(4)],
                   [Fraction(6), Fraction(8)]], object)
    with pytest.raises(ZeroDivisionError, match='orthogonalisable'):
        gram_schmidt(vs, sqrt=_exact_sqrt)

def test_gram_schmidt_with_sqrt_announces_the_normalising_cost(bars):
    #M*N divisions and M roots replace the M(M-1)/2 projection divisions,
    #while add, sub & mul stay exactly as they are without normalising
    gram_schmidt(_vs(3, 4), sqrt=math.sqrt, progress=True)
    assert {(b.desc, b.total) for b in bars.instances} \
            == {('add', 18), ('sub', 12), ('mul', 36),
                ('truediv', 12), ('sqrt', 3)}
    assert all(b.n == b.total for b in bars.instances)

def test_gram_schmidt_with_sqrt_fills_its_bars_exactly(bars):
    for _ in range(20):
        N = randint(1, 7)
        gram_schmidt(_vs(randint(0, N), N), sqrt=math.sqrt, progress=True)
    assert bars.instances and all(b.n == b.total for b in bars.instances)

def test_gram_schmidt_without_sqrt_announces_no_roots(bars):
    #the category only exists on the normalising path, so asking for it on the
    #other one has to be rejected like any untracked category
    gram_schmidt(_vs(3, 4), progress=True)
    assert 'sqrt' not in {b.desc for b in bars.instances}
    with pytest.raises(ValueError, match='untracked'):
        gram_schmidt(_vs(3, 4), progress=('sqrt',))



#edge cases
def test_gram_schmidt_of_no_vectors_does_nothing():
    vs = np.empty((0, 3))
    assert gram_schmidt(vs) == []
    assert vs.shape == (0, 3)

def test_gram_schmidt_of_a_single_vector_leaves_it_alone():
    vs = np.array([[3., 4.]])
    assert gram_schmidt(vs) == [25.]
    assert np.allclose(vs, [[3., 4.]])

def test_gram_schmidt_of_empty_vectors_raises():
    #a zero length vector has a zero norm, so it can never span anything
    with pytest.raises(ZeroDivisionError):
        gram_schmidt(np.empty((1, 0)))



#errors
def test_gram_schmidt_dependent_vectors_raise():
    with pytest.raises(ZeroDivisionError, match='orthogonalisable'):
        gram_schmidt(np.array([[1., 1.], [2., 2.]]))

@pytest.mark.parametrize('vs', [np.zeros(2), np.zeros((2, 2, 2))])
def test_gram_schmidt_errors_match_with_and_without_progress(bars, vs):
    #the sanitiser & announcer must not raise a different error first
    with pytest.raises(ValueError) as bare:
        gram_schmidt(vs)
    with pytest.raises(ValueError) as shown:
        gram_schmidt(vs, progress=True)
    assert str(bare.value) == str(shown.value)
