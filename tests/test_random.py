from linalg.random import *
from fractions import Fraction
from statistics import fmean, stdev
import numpy as np
import pytest



#############################################################################
#fully written by Claude - the rest of the suite samples its matrices from
#here, so a silent bug in `mrandqr` would weaken every test that uses it
#rather than fail one. the distribution tests are deliberately loose: the
#sample sizes put every bound at least ten standard errors away, so they
#pin the documented contract without ever failing on a bad draw.
#############################################################################

SAMPLES = 4000



#randz
def test_randz_returns_a_python_int():
    #not a numpy scalar - these fill object arrays that must do exact
    #Python arithmetic, with no silent promotion or overflow
    x = randz()
    assert type(x) is int

def test_randz_of_zero_sigma_is_always_zero():
    assert all(randz(0) == 0 for _ in range(50))

def test_randz_is_not_constant():
    assert len({randz() for _ in range(20)}) > 1

@pytest.mark.parametrize('sigma', [1, 10, 100])
def test_randz_stays_within_its_support(sigma):
    #binomialvariate(4s^2) - 2s^2 can only land in [-2s^2, +2s^2]
    assert all(abs(randz(sigma)) <= 2*sigma**2 for _ in range(200))

def test_randz_is_centred_on_zero():
    xs = [randz(10) for _ in range(SAMPLES)]
    assert abs(fmean(xs)) < 10/2

def test_randz_has_the_requested_standard_deviation():
    #the sigma argument is the standard deviation, not the binomial n
    for sigma in (10, 100):
        xs = [randz(sigma) for _ in range(SAMPLES)]
        assert abs(stdev(xs) - sigma) < sigma/10

def test_randz_rejects_a_negative_sigma():
    with pytest.raises(ValueError, match='non-negative'):
        randz(-1)



#randq
def test_randq_returns_a_fraction():
    assert isinstance(randq(), Fraction)

@pytest.mark.parametrize('grade', [1, 7, 1000])
def test_randq_denominator_divides_the_grade(grade):
    #randz(grade)/grade before cancelling, so the grade is the lattice spacing
    assert all(grade % randq(grade).denominator == 0 for _ in range(200))

@pytest.mark.parametrize('grade', [1, 7, 1000])
def test_randq_stays_within_its_support(grade):
    assert all(abs(randq(grade)) <= 2*grade for _ in range(200))

def test_randq_has_unit_variance_whatever_the_grade():
    #the grade refines the lattice, it does not widen the distribution -
    #easy to get backwards, and the reason `mrandqr` entries stay O(1)
    for grade in (7, 1000):
        xs = [float(randq(grade)) for _ in range(SAMPLES)]
        assert abs(stdev(xs) - 1) < 1/10

def test_randq_is_centred_on_zero():
    xs = [float(randq()) for _ in range(SAMPLES)]
    assert abs(fmean(xs)) < 1/2

@pytest.mark.parametrize('grade', [0, -1])
def test_randq_rejects_a_non_positive_grade(grade):
    #zero would divide by zero, so unlike sigma it may not be zero
    with pytest.raises(ValueError, match='positive'):
        randq(grade)



#vrandz & vrandq
@pytest.mark.parametrize('vrand', [vrandz, vrandq])
@pytest.mark.parametrize('shape, expected', [
    (1,      (1,)),
    (5,      (5,)),
    (0,      (0,)),
    ((2, 3), (2, 3)),
    ((2, 0), (2, 0)),
    ((2, 3, 4), (2, 3, 4)),
])
def test_vrand_shapes(vrand, shape, expected):
    assert vrand(shape).shape == expected

@pytest.mark.parametrize('vrand', [vrandz, vrandq])
def test_vrand_is_an_object_array(vrand):
    #object, so the scalars keep their exact Python arithmetic
    assert vrand((2, 3)).dtype == object

@pytest.mark.parametrize('vrand, scalar', [(vrandz, int), (vrandq, Fraction)])
def test_vrand_elements_are_scalars_of_the_right_type(vrand, scalar):
    assert all(isinstance(x, scalar) for x in vrand((3, 4)).flat)

@pytest.mark.parametrize('vrand', [vrandz, vrandq])
def test_vrand_draws_each_element_separately(vrand):
    assert len(set(vrand(30).flat)) > 1

def test_vrandz_forwards_its_sigma():
    #sigma=0 is the one deterministic draw, so it pins the forwarding
    assert np.all(vrandz(20, sigma=0) == 0)

def test_vrandq_forwards_its_grade():
    assert all(x.denominator == 1 for x in vrandq(20, grade=1).flat)

@pytest.mark.parametrize('vrand', [vrandz, vrandq])
def test_vrand_of_an_empty_shape_draws_nothing(vrand):
    assert vrand(0).size == 0



#mrandqr
@pytest.mark.parametrize('M, N', [(m, n)
        for m in range(1, 5+1) for n in range(1, 5+1)])
def test_mrandqr_has_the_requested_rank(M, N):
    #the contract the rest of the suite is built on: `test_rank_decomp`,
    #`test_det_gauss` and `test_ref_gauss` all trust this R
    for R in range(min(M, N)+1):
        A = mrandqr(M, N, R)
        assert np.linalg.matrix_rank(A.astype(float)) == R

@pytest.mark.parametrize('M, N, R', [(4, 3, 2), (3, 3, 0), (1, 6, 1)])
def test_mrandqr_shape_and_element_types(M, N, R):
    A = mrandqr(M, N, R)
    assert A.shape == (M, N) and A.dtype == object
    assert all(isinstance(x, Fraction) for x in A.flat)

def test_mrandqr_zeros_are_fractions_not_ints():
    #the whole reason for the rank zero branch: an empty object matmul
    #fills with int(0), which then poisons the exact arithmetic downstream
    A = mrandqr(2, 3, 0)
    assert all(type(x) is Fraction for x in A.flat)
    assert not np.any(A)

def test_mrandqr_rank_defaults_to_zero():
    assert not np.any(mrandqr(3, 3))

def test_mrandqr_forwards_its_grade():
    #entries are sums of products of two draws, so the denominators
    #divide grade squared rather than grade
    A = mrandqr(3, 4, 2, grade=6)
    assert all(6**2 % x.denominator == 0 for x in A.flat)

@pytest.mark.parametrize('M, N, R', [(0, 3, 0), (3, 0, 0), (0, 0, 0),
                                     (0, 3, 2), (3, 0, 2)])
def test_mrandqr_of_a_degenerate_shape(M, N, R):
    A = mrandqr(M, N, R)
    assert A.shape == (M, N) and A.dtype == object

def test_mrandqr_is_not_constant():
    assert not np.array_equal(mrandqr(3, 3, 3), mrandqr(3, 3, 3))



#exports
def test_the_package_re_exports_the_samplers():
    #`linalg/__init__.py` does `from .random import *`
    import linalg
    from linalg.random import __all__ as names
    assert all(hasattr(linalg, name) for name in names)
