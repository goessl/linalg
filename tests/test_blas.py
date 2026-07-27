from linalg import *
import numpy as np
from functools import partial
from itertools import permutations
from sys import float_info
import pytest



#preparing values
nive_values = (
    np.array([1, 2, 3], dtype=np.int64),
    np.array([[5, 6, 7],
              [8, 9, 10]], dtype=np.float64),
    7,
    8.0
)

#integers
#for unsigned & signed
#for Python & numpy types
#their boundaries & one over
WIDTHS = (8, 16, 32, 64)
py_uint = tuple(sorted({
        v for w in WIDTHS
          for v in (0, 2**w-1, 2**w)
}))
assert len(py_uint) == 9
py_int  = tuple(sorted({
        v for w in WIDTHS
          for v in (-2**(w-1)-1, -2**(w-1), +2**(w-1)-1, +2**(w-1))
}))
assert len(py_int) == 16

UINTS = (np.uint8, np.uint16, np.uint32, np.uint64)
INTS = (np.int8, np.int16, np.int32, np.int64)
np_uint = tuple(
        t(v) for t in UINTS
             for v in (np.iinfo(t).min, np.iinfo(t).max)
)
assert len(np_uint) == 8
np_int  = tuple(
        t(v) for t in INTS
             for v in (np.iinfo(t).min, np.iinfo(t).max)
)
assert len(np_int) == 8

#floats
#for Python & numpy types
#signed zeros, the specials, and per dtype the two smallest
#magnitudes, the largest, and one past it, which overflows that dtype to inf.
py_float = (
        0.0, -0.0, -1.0, +1.0,
        float_info.min, float_info.max,
        -float_info.epsilon, +float_info.epsilon,
        float('nan'), float('inf'), float('-inf')
)

FLOATS = (np.float16, np.float32, np.float64)
np_float = tuple(v for t in FLOATS
                   for v in (t(-0.0), t(+0.0),
                             -np.finfo(t).smallest_normal, +np.finfo(t).smallest_normal,
                             -np.finfo(t).smallest_subnormal, +np.finfo(t).smallest_subnormal,
                             -np.finfo(t).max, +np.finfo(t).max,
                             t(-float('inf')), t(+float('inf')), t(-float('nan')))
)

#arrays
nasty_arrays = (
    np.array([np.iinfo(np.int8).min,  0, np.iinfo(np.int8).max],  np.int8),
    np.array([0, 1, np.iinfo(np.uint8).max],                      np.uint8),
    np.array([np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max], np.int64),
    np.array([0, 1, np.iinfo(np.uint64).max],                     np.uint64),
    np.array([-np.inf, -0.0, np.nan],                             np.float16),
    np.array([np.finfo(np.float32).max, 0.0, np.inf],             np.float32),
    np.array([np.finfo(np.float64).smallest_subnormal, -0.0, np.nan], np.float64),
    np.array([complex(np.nan, 0), complex(0, np.inf), 0j],        np.complex128),
    np.array([True, False, True]),
    
    #object arrays: NumPy unwraps the elements to Python objects, and they are
    #the only arrays that can hold an integer no fixed width fits
    np.array([1, 2, 3],           object),
    np.array([2**70, -2**70, 0],  object),
    
    #0-d, which is what makes a result a scalar rather than an array
    np.array(0), np.array(-0.0), np.array(7, object),
    
    #zero-size, and a shape that broadcasts against the (3,) arrays above
    np.empty((0, 3)),
    np.ones((2, 1)),
    
    #not C-contiguous, the allocated result should follow the iteration order
    np.asfortranarray(np.arange(6.).reshape(2, 3)),
    np.arange(6)[::2],
)


values = nive_values + py_uint + py_int + np_uint + np_int + py_float + np_float + nasty_arrays



#supervising functions
def _compare_results(x, y):
    """Return if two results are equal.
    
    They must both be scalars, `numpy.ndarray`s or `tuple`s.
    Scalars must have the same `type` and have the same `repr`.
    `numpy.ndarray`s must have the same `dtype` and  have the same `repr`.
    `tuple`s, are compared elementwise recursively.
    
    Notes
    -----
    Comparison via `repr` because `nan==nan` is false.
    """
    if isinstance(x, tuple) and isinstance(y, tuple):
        return len(x)==len(y) and all(map(_compare_results, x, y))
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return x.dtype==y.dtype and repr(x.tolist())==repr(y.tolist())
    elif not isinstance(x, np.ndarray) and not isinstance(y, np.ndarray):
        return type(x)==type(y) and repr(x)==repr(y)
    return False #different types

def _compare_functions(f, g, *args, **kwargs):
    """Return if two functions behave the same.
    
    They must either throw the same or return the same.
    """
    try:
        f_res = f(*args, **kwargs)
    except Exception as ex:
        with pytest.raises(type(ex)):
            g(*args, **kwargs)
        return True
    g_res = g(*args, **kwargs)
    return _compare_results(f_res, g_res)



#tests
UNARY  = (np.positive, np.negative)
BINARY = (np.add, np.subtract, np.multiply,
          np.divide, np.floor_divide, np.mod)
MULTI_UNARY  = (np.modf, np.frexp)
MULTI_BINARY = (np.divmod, )

@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('op', UNARY+MULTI_UNARY)
def test_unary(op):
    for a in values:
        assert _compare_functions(op, partial(ufunc_with_cb, op), a)

@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('op', BINARY+MULTI_BINARY)
def test_binary(op):
    for a, b in permutations(values, 2):
        assert _compare_functions(op, partial(ufunc_with_cb, op), a, b)


#problematic promotions from https://numpy.org/doc/stable/reference/arrays.promotion.html
def test_promotion():
    add = partial(ufunc_with_cb, np.add)
    mul = partial(ufunc_with_cb, np.multiply)
    
    assert _compare_results(
            add(np.int8(1), np.int8(1)),
            np.int8(2))
    
    assert _compare_results(
            add(np.int8(4), np.int64(8)),
            np.int64(12))
    assert _compare_results(
            add(np.float32(3), np.float16(3)),
            np.float32(6))
    
    assert _compare_results(
            add(np.array([1, 2.5, 2.1], dtype=np.float32), 10.0),
            np.array([11. , 12.5, 12.1], dtype=np.float32))
    assert _compare_results(
            add(np.array([3, 5, 7], dtype=np.int16), 10),
            np.array([13, 15, 17], dtype=np.int16))
    
    with pytest.raises(OverflowError):
        add(np.int8(1), 1000)
    with pytest.raises(OverflowError):
        mul(np.int64(1), 10**100)
    with pytest.warns(RuntimeWarning):
        add(np.float32(1), 1e300)
    
    #with pytest.warns(RuntimeWarning):
    #    add(np.int8(100), 100)
    assert _compare_results(
            add(np.array(100, dtype=np.int8), 100),
            np.int8(-56))



#- - - Claude tests below - - -

#NEP 50 weak scalar promotion

i8  = np.array([1, 2], dtype=np.int8)
u8  = np.array([1, 2], dtype=np.uint8)
f16 = np.array([1, 2], dtype=np.float16)
f32 = np.array([1, 2], dtype=np.float32)
c64 = np.array([1+2j],  dtype=np.complex64)
bl  = np.array([True, False])
ob  = np.array([1, 2], dtype=object)


def outcome(f, *args):
    """Dtype & values of `f(*args)`, or the class of the exception it raises,
    so `ufunc_with_cb` and the bare ufunc can be compared directly.
    NumPy's `UFuncTypeError`s are normalised to their `TypeError` base."""
    try:
        result = np.asarray(f(*args))
    except OverflowError:
        return OverflowError
    except TypeError:
        return TypeError
    except ValueError:
        return ValueError
    return result.dtype, result.tolist()


@pytest.mark.parametrize('x', (7, 7.0, 1j, True, [1, 2], i8, f32, ob))
@pytest.mark.parametrize('op', UNARY)
def test_unary_promotion(op, x):
    assert outcome(partial(ufunc_with_cb, op), x) == outcome(op, x)

@pytest.mark.parametrize('x, y', (
    (i8, 1), (i8, 2.0), (i8, True),  #weak scalars adopt the narrow array dtype
    (f16, 3.0), (f32, 2), (f32, 2.0),
    (c64, 2), (c64, 1j), (bl, 1), (ob, 1),
    (7, 7), (7, 7.0), (True, True), ([1, 2], 7),  #no array operand at all
    (i8, f32), (f16, f32)                         #two strong operands
))
@pytest.mark.parametrize('op', BINARY)
def test_binary_promotion(op, x, y):
    assert outcome(partial(ufunc_with_cb, op), x, y) == outcome(op, x, y)

#examples from the NumPy data type promotion reference
#https://numpy.org/doc/stable/reference/arrays.promotion.html
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('op, x, y, dtype', (
    (np.add, np.int8(1), np.int8(1), np.int8),
    (np.add, np.int8(4), np.int64(8), np.int64),         #64 > 8
    (np.add, np.float32(3), np.float16(3), np.float32),  #32 > 16
    #a weak scalar contributes its kind, but not its precision
    (np.add, np.array([1, 2.5, 2.1], dtype=np.float32), 10.0, np.float32),
    (np.add, np.array([3, 5, 7], dtype=np.int16), 10, np.int16),
    (np.add, np.int16(1), 1.0, np.float64),
    (np.add, np.array([True, False]), 1.0, np.float64),
    #a weak scalar may overflow the dtype it adopts
    (np.add, np.float32(1), 1e300, np.float32),          #-> inf
    (np.add, np.int8(100), 100, np.int8),                #-> -56
    #no wider integer exists, and integer precision forces a wider float
    (np.add, np.int64(1), np.uint64(1), np.float64),
    (np.add, np.int64(1), np.float16(1), np.float64),
    (np.add, np.uint8(1), np.int8(1), np.int16),
    #division always returns floating point
    (np.divide, np.array([3, 5, 7], dtype=np.int16), 2, np.float64)
))
def test_promotion_reference(op, x, y, dtype):
    assert outcome(partial(ufunc_with_cb, op), x, y) == outcome(op, x, y)
    assert ufunc_with_cb(op, x, y).dtype == dtype

#a weak scalar that does not fit the dtype the loop resolved to
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('op, x, y', (
    (np.add, u8, 1),          #int64 does not cast to uint8 with `same_kind`
    (np.add, f32, 2**70),     #no fixed-width dtype fits, so `object`
    (np.divide, f16, 1e6),    #would be computed in float64, rounded twice
    #the two reference examples that raise `OverflowError` in NumPy
    (np.add, np.int8(1), 1000),
    (np.multiply, np.int64(1), 10**100)
))
def test_binary_promotion_out_of_range(op, x, y):
    assert outcome(partial(ufunc_with_cb, op), x, y) == outcome(op, x, y)

#a lone weak scalar must not be typed weakly: `numpy.positive(2**64)` resolves
#to `object`, while `int` would resolve to `int64` and overflow
@pytest.mark.parametrize('op', UNARY)
@pytest.mark.parametrize('x', (2**64, 2**70, -2**63-1))
def test_unary_weak_scalar_too_big(op, x):
    assert outcome(partial(ufunc_with_cb, op), x) == outcome(op, x)
    #`object` is the only dtype that fits, and a 0-d `object` result comes
    #back as the Python object itself, not as a `numpy` scalar
    assert type(ufunc_with_cb(op, x)) is int



#the results are allocated by `numpy.nditer`, not by the ufunc, so their
#shape, their memory order, and whether they are arrays at all are worth
#checking

def _as_tuple(result, op):
    """Return `result` as a `tuple`, so single- & multi-output ufuncs can be
    handled the same way."""
    return result if op.nout>1 else (result, )


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('op', UNARY+BINARY+MULTI_UNARY+MULTI_BINARY)
def test_result_layout(op):
    for operands in permutations(values, op.nin):
        try:
            expected = op(*operands)
        except Exception:
            continue
        result = ufunc_with_cb(op, *operands)
        assert len(_as_tuple(result, op)) == op.nout
        for r, e in zip(_as_tuple(result, op), _as_tuple(expected, op)):
            assert type(r) is type(e)
            assert np.shape(r) == np.shape(e)
            if isinstance(e, np.ndarray):
                assert r.flags.c_contiguous == e.flags.c_contiguous
                assert r.flags.f_contiguous == e.flags.f_contiguous

def test_result_is_a_scalar_only_for_0d():
    #0-d operands give back a `numpy` scalar, like the ufunc itself does
    assert isinstance(ufunc_with_cb(np.add, 7, 3), np.generic)
    assert isinstance(ufunc_with_cb(np.add, np.array(7), np.array(3)), np.generic)
    #anything with a shape, including a 1-element or an empty array, stays one
    assert isinstance(ufunc_with_cb(np.add, np.array([7]), 3), np.ndarray)
    assert ufunc_with_cb(np.add, np.empty((0, 3)), 1.0).shape == (0, 3)
    assert ufunc_with_cb(np.add, np.ones((2, 1)), np.arange(3)).shape == (2, 3)

def test_result_is_writeable_and_owns_its_data():
    result = ufunc_with_cb(np.add, np.ones(3), 1)
    assert result.flags.writeable and result.flags.owndata





#`cb` is the only thing `ufunc_with_cb` adds to a plain ufunc call, so how
#often it fires, with how many arguments, in which order and with which
#Python types is tested on its own

def _record(op, *operands):
    """Return the calls `cb` receives, and the result, of `op(*operands)`."""
    calls = []
    result = ufunc_with_cb(op, *operands, cb=lambda *args: calls.append(args))
    return calls, result


CB_OPERANDS = {
    1: (
        (np.array([1, 2, 3]),),
        (np.array([[1.0, 2.0], [3.0, 4.0]]),),
        (np.array(7),),
        (7,),
        (np.zeros((0, 3)),),                              #empty
        (np.asfortranarray(np.arange(6.).reshape(2, 3)),),#F-order
        (np.arange(6)[::2],),                             #strided
        (np.array([1, 2], object),),
    ),
    2: (
        (np.array([1, 2, 3]), 7),
        (np.array([1, 2, 3]), np.ones((2, 1))),           #broadcast
        (np.array(7), np.array(3)),
        (7, 3),
        (np.zeros((0, 3)), 1.0),                          #empty
        (np.asfortranarray(np.arange(6.).reshape(2, 3)), 1.0),
        (np.arange(6)[::2], np.int8(2)),
        (np.array([1, 2], object), 1),
    )
}


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('op', UNARY+BINARY+MULTI_UNARY+MULTI_BINARY)
def test_cb_fires_once_per_result_element(op):
    for operands in CB_OPERANDS[op.nin]:
        try:
            op(*operands) #not every op accepts every operand, e.g. `modf(int)`
        except Exception:
            continue
        calls, result = _record(op, *operands)
        assert len(calls) == np.broadcast(*operands).size
        assert all(len(calls)==np.asarray(r).size
                   for r in _as_tuple(result, op))

@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('op', UNARY+BINARY+MULTI_UNARY+MULTI_BINARY)
def test_cb_gets_the_operands_and_the_results(op):
    #`op.nargs` values per call, the `op.nin` operands first, then every one
    #of the `op.nout` results
    for operands in CB_OPERANDS[op.nin]:
        try:
            op(*operands)
        except Exception:
            continue
        calls, result = _record(op, *operands)
        assert all(len(call)==op.nargs for call in calls)
        #`numpy.nditer` picks the iteration order, which is `ravel`'s 'K'
        for i, r in enumerate(_as_tuple(result, op)):
            assert repr([call[op.nin+i] for call in calls]) \
                    == repr(np.asarray(r).ravel(order='K').tolist())

def test_cb_gets_python_scalars_of_the_resolved_dtype():
    #the operands arrive in the dtype the loop resolved to, not their own,
    #so `int / int` reports floats. `==` alone would not catch that, as 1 == 1.0.
    calls, _ = _record(np.divide, np.array([1, 2, 3]), 7)
    assert calls == [(1.0, 7.0, 1/7), (2.0, 7.0, 2/7), (3.0, 7.0, 3/7)]
    assert all(type(v) is float for call in calls for v in call)

    calls, _ = _record(np.negative, np.array([1, 2, 3]))
    assert calls == [(1, -1), (2, -2), (3, -3)]
    assert all(type(v) is int for call in calls for v in call)

    #a broadcast operand is widened to the resolved dtype & repeated
    calls, _ = _record(np.add, np.array([1, 2, 3]),
                               np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]))
    assert len(calls) == 6
    assert calls[0] == (1.0, 5.0, 6.0)
    assert all(type(v) is float for call in calls for v in call)

    #bool & complex reach the callback as the Python types too
    calls, _ = _record(np.multiply, np.array([1+2j]), 2)
    assert all(type(v) is complex for call in calls for v in call)
    calls, _ = _record(np.multiply, np.array([True, False]), True)
    assert all(type(v) is bool for call in calls for v in call)

    #an object array hands its Python objects through untouched, so an integer
    #no fixed-width dtype could hold survives
    calls, _ = _record(np.add, np.array([2**70], object), 1)
    assert calls == [(2**70, 1, 2**70+1)]

def test_cb_not_called_for_an_empty_result():
    calls, result = _record(np.add, np.zeros((0, 3)), 1.0)
    assert calls == [] and result.shape == (0, 3)

def test_cb_called_once_for_scalars():
    assert _record(np.add, 7, 3)[0] == [(7, 3, 10)]
    assert _record(np.negative, np.array(7))[0] == [(7, -7)]

def test_cb_exception_propagates():
    with pytest.raises(ZeroDivisionError):
        ufunc_with_cb(np.add, np.ones(3), 1, cb=lambda *args: 1/0)
    #the iterator has to be closed cleanly, so later calls still work
    assert _compare_results(ufunc_with_cb(np.add, 7, 3), np.add(7, 3))

def test_cb_is_keyword_only():
    #a positional argument is an operand, so `numpy.add` gets three of them
    with pytest.raises(TypeError):
        ufunc_with_cb(np.add, 1, 2, lambda *args: None)





#multi-output specifics, on top of the differential tests above

def test_multi_output_returns_one_result_per_output():
    #0-d operands make every output a scalar, not a 0-d array
    q, r = ufunc_with_cb(np.divmod, 7, 3)
    assert (type(q), type(r)) == (np.int64, np.int64)
    assert (q, r) == (2, 1)
    #the outputs don't have to share a dtype
    m, e = ufunc_with_cb(np.frexp, np.float16(1.5))
    assert (m.dtype, e.dtype) == (np.float16, np.intc)
    assert (m, e) == (0.75, 1)
    #an empty operand gives `op.nout` empty results, not an empty `tuple`
    assert [r.shape for r in ufunc_with_cb(np.modf, np.zeros((0, 3)))] \
            == [(0, 3), (0, 3)]

def test_multi_output_cb():
    #the callback gets the operands *and* every output, so `op.nargs` values
    assert _record(np.divmod, np.array([7, 8]), 3)[0] \
            == [(7, 3, 2, 1), (8, 3, 2, 2)]

    #and they arrive as the Python types of the dtype each output resolved to
    calls, _ = _record(np.frexp, np.array([1.5, 2.5]))
    assert calls == [(1.5, 0.75, 1), (2.5, 0.625, 2)]
    assert all(tuple(map(type, call))==(float, float, int) for call in calls)

def test_multi_output_weak_promotion():
    #a weak scalar adopts the array dtype for every output
    q, r = ufunc_with_cb(np.divmod, np.array([7, 8], np.int8), 3)
    assert q.dtype == r.dtype == np.int8
    #and may fail to, exactly like `numpy` itself
    with pytest.raises(OverflowError):
        ufunc_with_cb(np.divmod, np.array([7], np.uint8), 300)


def test_subclasses_not_preserved():
    #`numpy.nditer` allocates a plain `numpy.ndarray`, so a subclass, and with
    #it a `numpy.ma` mask, is lost, while `numpy` itself would keep it
    masked = np.ma.array([1, 2, 3], mask=[0, 1, 0])
    assert type(np.add(masked, 1)) is np.ma.MaskedArray
    assert type(ufunc_with_cb(np.add, masked, 1)) is np.ndarray
