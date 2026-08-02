"""Vectorial operations.

Some inbetween of

- [BLAS](https://netlib.org/blas/),
- [Python operations](https://docs.python.org/3/library/operator.html) &
- [`numpy` functions](https://numpy.org/doc/stable/reference/ufuncs.html).

Provides

- [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]

    run a `numpy.ufunc`
    with a callback after every scalar operation

- [`vpos`][linalg.blas.vpos], [`vneg`][linalg.blas.vneg],
[`vadd`][linalg.blas.vadd], [`vsub`][linalg.blas.vsub],
[`vmul`][linalg.blas.vmul], [`vtruediv`][linalg.blas.vtruediv],
[`vfloordiv`][linalg.blas.vfloordiv], [`vmod`][linalg.blas.vmod]

    vectorised operations on `npt.ArrayLike`s with visualisation support
"""



from .progress import visualisable, notify
import numpy as np
import numpy.typing as npt
from typing import Any
from collections.abc import Callable



__all__ = (
    'ufunc_with_cb',
    'vpos', 'vneg',
    'vadd', 'vsub',
    'vmul', 'vtruediv', 'vfloordiv', 'vmod'
)



def _is_weak(x: Any) -> bool:
    """Return whether `x` is a [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html) weak scalar.
    
    - Python `int`, `float` or `complex` are weak.
    - All others (`bool`, `numpy.int64`, ...) are strong.
    
    References
    ----------
    - [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)
    """
    return (isinstance(x, (int, float, complex))
            and not isinstance(x, (bool, np.generic)))

def _weak_dtype(x: npt.ArrayLike) -> Any:
    """Return the `type` (if weak) or `dtype` (if strong) of `x`.
    
    References
    ----------
    - [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)
    """
    return type(x) if _is_weak(x) else np.asarray(x).dtype

def ufunc_with_cb(
    op:        np.ufunc,
    *operands: npt.ArrayLike,
    cb:        Callable[..., None] | None = None
) -> Any:
    """Apply a [`numpy.ufunc`](https://numpy.org/doc/stable/reference/ufuncs.html) element-wise, with an optional callback.
    
    Same as calling `op` directly, but with an additional callback parameter.
    If set, gets called after every scalar operation with the scalar operands
    and the scalar results.
    
    The behaviour (broadcasting, promotion, throws, ...) should be exactly
    the same as calling `op` on its own.
    Whole complexity is just for this behaviour conservation.
    
    Tested for operations:
    
    - [`numpy.positive`](https://numpy.org/doc/stable/reference/generated/numpy.positive.html)
    - [`numpy.negative`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html)
    - [`numpy.add`](https://numpy.org/doc/stable/reference/generated/numpy.add.html)
    - [`numpy.subtract`](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html)
    - [`numpy.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)
    - [`numpy.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html)
    - [`numpy.floor_divide`](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html)
    - [`numpy.mod`](https://numpy.org/doc/stable/reference/generated/numpy.mod.html)
    - [`numpy.divmod`](https://numpy.org/doc/stable/reference/generated/numpy.divmod.html)
    - [`numpy.modf`](https://numpy.org/doc/stable/reference/generated/numpy.modf.html)
    - [`numpy.frexp`](https://numpy.org/doc/stable/reference/generated/numpy.frexp.html)
    
    Not yet correctly behaving for [`numpy.ma.MaskedArray`](https://numpy.org/doc/stable/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray).
    
    Improved code from the [`numpy.nditer`](https://numpy.org/doc/stable/reference/generated/numpy.nditer.html) examples.
    
    Obviously really slow.
    
    Notes
    -----
    Naive versions that not perfectly imitate ufunc promotion:
    ```python
    def unary(op:Callable[[Any], Any], a:npt.NDArray)
            -> npt.NDArray:
        a = np.asarray(a)
        r = np.empty(a.shape, a.dtype)
        for ai, ri in np.nditer(
            [a, r],
            flags = ['refs_ok', 'zerosize_ok'],
            op_flags = [['readonly'], ['writeonly']]
        ):
            ri[...] = op(ai.item())
        return r
    
    def binary(op:Callable[[Any, Any], Any], a:npt.NDArray, b:npt.NDArray)
            -> npt.NDArray:
        a, b = np.asarray(a), np.asarray(b)
        r = np.empty(np.broadcast_shapes(a.shape, b.shape),
                     np.result_type(a.dtype, b.dtype))
        for ai, bi, ri in np.nditer(
            [a, b, r],
            flags = ['refs_ok', 'zerosize_ok'],
            op_flags = [['readonly'], ['readonly'], ['writeonly']]
        ):
            ri[...] = op(ai.item(), bi.item())
        return r
    ```
    
    References
    ----------
    - [`numpy.ufunc`](https://numpy.org/doc/stable/reference/ufuncs.html)
    - [`numpy.nditer`](https://numpy.org/doc/stable/reference/generated/numpy.nditer.html)
    - [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)
    """
    if len(operands) > 1:
        dtypes = op.resolve_dtypes(tuple(_weak_dtype(o) for o in operands)
                                   + op.nout*(None,))
    else: #single arguments need some dtype to clinge onto
        dtypes = op.resolve_dtypes(tuple(np.asarray(o).dtype for o in operands)
                                   + op.nout*(None,))
    
    it = np.nditer(
        tuple(np.asarray(o, t if _is_weak(o) else None)
              for o, t in zip(operands, dtypes)) + op.nout*(None,),
        flags = ['buffered', 'refs_ok', 'zerosize_ok'],
        op_flags = op.nin * [['readonly']]
                 + op.nout * [['writeonly', 'allocate']],
        op_dtypes = dtypes,
        casting = 'unsafe' #dtype safely determined by ufunc itself
    )
    with it:
        for args in it:
            op(*args[:op.nin], out=args[-op.nout:])
            if cb is not None:
                cb(*(a.item() for a in args))
        result = it.operands[-op.nout:]
        #possibly unpack
        result = tuple(r if bool(r.ndim) else r[()] for r in result)
        return result[0] if op.nout==1 else result



@visualisable(lambda a: {'pos':np.broadcast(a).size})
def vpos(a: npt.ArrayLike) -> npt.NDArray:
    """Return the elementwise affirmation.
    
    Exactly like [`numpy.positive`](https://numpy.org/doc/stable/reference/generated/numpy.positive.html).
    
    Complexity
    ----------
    For an array with `n` elements there will be
    
    - `n` many scalar 'positives' (`pos`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    
    References
    ----------
    - [`numpy.positive`](https://numpy.org/doc/stable/reference/generated/numpy.positive.html)
    """
    return ufunc_with_cb(np.positive, a,
            cb=lambda *_: notify('pos'))

@visualisable(lambda a: {'neg':np.broadcast(a).size})
def vneg(a: npt.ArrayLike) -> npt.NDArray:
    """Return the elementwise negative.
    
    Exactly like [`numpy.negative`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html).
    
    Complexity
    ----------
    For an array with `n` elements there will be
    
    - `n` many scalar negations (`neg`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    
    References
    ----------
    - [`numpy.negative`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html)
    """
    return ufunc_with_cb(np.negative, a,
            cb=lambda *_: notify('neg'))

@visualisable(lambda a, b: {'add': np.broadcast(a, b).size})
def vadd(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    """Return the elementwise sum.
    
    Exactly like [`numpy.add`](https://numpy.org/doc/stable/reference/generated/numpy.add.html).
    
    Complexity
    ----------
    For a resulting array with `n` elements there will be
    
    - `n` many scalar additions (`add`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    
    References
    ----------
    - [`numpy.add`](https://numpy.org/doc/stable/reference/generated/numpy.add.html)
    """
    return ufunc_with_cb(np.add, a, b,
            cb=lambda *_: notify('add'))

@visualisable(lambda a, b: {'sub': np.broadcast(a, b).size})
def vsub(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    """Return the elementwise difference.
    
    Exactly like [`numpy.subtract`](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html).
    
    Complexity
    ----------
    For a resulting array with `n` elements there will be
    
    - `n` many scalar subtractions (`sub`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    
    References
    ----------
    - [`numpy.subtract`](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html)
    """
    return ufunc_with_cb(np.subtract, a, b,
            cb=lambda *_: notify('sub'))

@visualisable(lambda a, b: {'mul': np.broadcast(a, b).size})
def vmul(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    """Return the elementwise product.
    
    Exactly like [`numpy.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html).
    
    Complexity
    ----------
    For a resulting array with `n` elements there will be
    
    - `n` many scalar multiplications (`mul`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    
    References
    ----------
    - [`numpy.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)
    """
    return ufunc_with_cb(np.multiply, a, b,
            cb=lambda *_: notify('mul'))

@visualisable(lambda a, b: {'truediv': np.broadcast(a, b).size})
def vtruediv(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    """Return the elementwise true quotient.
    
    Exactly like [`numpy.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html).
    
    Complexity
    ----------
    For a resulting array with `n` elements there will be
    
    - `n` many scalar true divisions (`truediv`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    
    References
    ----------
    - [`numpy.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html)
    """
    return ufunc_with_cb(np.divide, a, b,
            cb=lambda *_: notify('truediv'))

@visualisable(lambda a, b: {'floordiv': np.broadcast(a, b).size})
def vfloordiv(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    """Return the elementwise floored quotient.
    
    Exactly like [`numpy.floor_divide`](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html).
    
    Complexity
    ----------
    For a resulting array with `n` elements there will be
    
    - `n` many scalar floor divisions (`floordiv`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    
    References
    ----------
    - [`numpy.floor_divide`](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html)
    """
    return ufunc_with_cb(np.floor_divide, a, b,
            cb=lambda *_: notify('floordiv'))

@visualisable(lambda a, b: {'mod': np.broadcast(a, b).size})
def vmod(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray:
    """Return the elementwise remainder.
    
    Exactly like [`numpy.mod`](https://numpy.org/doc/stable/reference/generated/numpy.mod.html).
    
    Complexity
    ----------
    For a resulting array with `n` elements there will be
    
    - `n` many scalar mod operations (`mod`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    
    References
    ----------
    - [`numpy.mod`](https://numpy.org/doc/stable/reference/generated/numpy.mod.html)
    """
    return ufunc_with_cb(np.mod, a, b,
            cb=lambda *_: notify('mod'))
