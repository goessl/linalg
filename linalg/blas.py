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

    vectorised operations on `numpy.typing.ArrayLike`s
    with visualisation support
"""



from .progress import Progress, visualisable
import numpy as np
from numpy.typing import ArrayLike
from typing import Any
from collections.abc import Callable



__all__ = (
    'ufunc_with_cb',
    'vpos_announce', 'vpos',
    'vneg_announce', 'vneg',
    'vadd_announce', 'vadd',
    'vsub_announce', 'vsub',
    'vmul_announce', 'vmul',
    'vtruediv_announce', 'vtruediv',
    'vfloordiv_announce', 'vfloordiv',
    'vmod_announce', 'vmod'
)



def _is_weak(x: Any) -> bool:
    """Return whether `x` is a [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html) weak scalar.
    
    - Python `int`, `float` or `complex` are weak.
    - All others (`bool`, `numpy.int64`, ...) are strong.
    
    Parameters
    ----------
    x : Any
        Test object.
    
    Returns
    -------
    bool
        Whether `x` is a [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)
        weak scalar.
    
    References
    ----------
    - [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)
    """
    return (isinstance(x, (int, float, complex))
            and not isinstance(x, (bool, np.generic)))

def _weak_dtype(x: ArrayLike) -> Any:
    """Return the `type` (if weak) or `dtype` (if strong) of `x`.
    
    Parameters
    ----------
    x : numpy.typing.ArrayLike
        Test object.
    
    Returns
    -------
    type|dtype
        `type` or `dtype` of `x`.
    
    References
    ----------
    - [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)
    """
    return type(x) if _is_weak(x) else np.asarray(x).dtype

def ufunc_with_cb(op: np.ufunc, *operands: ArrayLike,
        cb: Callable[...,None]|None=None) -> Any:
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
    
    Parameters
    ----------
    op : numpy.ufunc
        Operation.
    operands : numpy.typing.ArrayLike
        Operands.
    cb : Callable[...,None]|None = None
        Callback.
    
    Returns
    -------
    Any
        Result.
    
    Notes
    -----
    Naive versions that not perfectly imitate ufunc promotion:
    ```python
    def unary(op:Callable[[Any], Any], a:NDArray)
            -> NDArray:
        a = np.asarray(a)
        r = np.empty(a.shape, a.dtype)
        for ai, ri in np.nditer(
            [a, r],
            flags = ['refs_ok', 'zerosize_ok'],
            op_flags = [['readonly'], ['writeonly']]
        ):
            ri[...] = op(ai.item())
        return r
    
    def binary(op:Callable[[Any, Any], Any], a:NDArray, b:NDArray)
            -> NDArray:
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



def vpos_announce(a: ArrayLike) -> dict[str,int]:
    """`vpos` announcer.
    
    See also
    --------
    - [`vpos`][linalg.blas.vpos]
    """
    return {'pos': np.size(a)}

@visualisable(vpos_announce)
def vpos(a: ArrayLike, *, progress: Progress) -> Any:
    """Return the elementwise affirmation.
    
    Exactly like [`numpy.positive`](https://numpy.org/doc/stable/reference/generated/numpy.positive.html).
    
    Parameters
    ----------
    a : numpy.typing.ArrayLike
        Input.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    Any
        Affirmation. An array for array like operands, a scalar otherwise.
    
    Complexity
    ----------
    For an array with `n` elements there will be
    
    - `n` many scalar 'positives' (`pos`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    - [`vpos_announce`][linalg.blas.vpos_announce]
    
    References
    ----------
    - [`numpy.positive`](https://numpy.org/doc/stable/reference/generated/numpy.positive.html)
    """
    return ufunc_with_cb(np.positive, a,
            cb=lambda *_: progress.update('pos'))

def vneg_announce(a: ArrayLike) -> dict[str,int]:
    """`vneg` announcer.
    
    See also
    --------
    - [`vneg`][linalg.blas.vneg]
    """
    return {'neg': np.size(a)}

@visualisable(vneg_announce)
def vneg(a: ArrayLike, *, progress: Progress) -> Any:
    """Return the elementwise negative.
    
    Exactly like [`numpy.negative`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html).
    
    Parameters
    ----------
    a : numpy.typing.ArrayLike
        Input.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    Any
        Negation. An array for array like operands, a scalar otherwise.
    
    Complexity
    ----------
    For an array with `n` elements there will be
    
    - `n` many scalar negations (`neg`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    - [`vneg_announce`][linalg.blas.vneg_announce]
    
    References
    ----------
    - [`numpy.negative`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html)
    """
    return ufunc_with_cb(np.negative, a,
            cb=lambda *_: progress.update('neg'))

def vadd_announce(a: ArrayLike, b: ArrayLike) -> dict[str,int]:
    """`vadd` announcer.
    
    See also
    --------
    - [`vadd`][linalg.blas.vadd]
    """
    return {'add': np.broadcast(a, b).size}

@visualisable(vadd_announce)
def vadd(a: ArrayLike, b: ArrayLike, *, progress: Progress) -> Any:
    """Return the elementwise sum.
    
    Exactly like [`numpy.add`](https://numpy.org/doc/stable/reference/generated/numpy.add.html).
    
    Parameters
    ----------
    a, b : numpy.typing.ArrayLike
        Inputs.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    Any
        Sum. An array for array like operands, a scalar otherwise.
    
    Complexity
    ----------
    For a resulting array with `n` elements there will be
    
    - `n` many scalar additions (`add`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    - [`vadd_announce`][linalg.blas.vadd_announce]
    
    References
    ----------
    - [`numpy.add`](https://numpy.org/doc/stable/reference/generated/numpy.add.html)
    """
    return ufunc_with_cb(np.add, a, b,
            cb=lambda *_: progress.update('add'))

def vsub_announce(a: ArrayLike, b: ArrayLike) -> dict[str,int]:
    """`vsub` announcer.
    
    See also
    --------
    - [`vsub`][linalg.blas.vsub]
    """
    return {'sub': np.broadcast(a, b).size}

@visualisable(vsub_announce)
def vsub(a: ArrayLike, b: ArrayLike, *, progress: Progress) -> Any:
    """Return the elementwise difference.
    
    Exactly like [`numpy.subtract`](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html).
    
    Parameters
    ----------
    a, b : numpy.typing.ArrayLike
        Inputs.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    Any
        Difference. An array for array like operands, a scalar otherwise.
    
    Complexity
    ----------
    For a resulting array with `n` elements there will be
    
    - `n` many scalar subtractions (`sub`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    - [`vsub_announce`][linalg.blas.vsub_announce]
    
    References
    ----------
    - [`numpy.subtract`](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html)
    """
    return ufunc_with_cb(np.subtract, a, b,
            cb=lambda *_: progress.update('sub'))

def vmul_announce(a: ArrayLike, b: ArrayLike) -> dict[str,int]:
    """`vmul` announcer.
    
    See also
    --------
    - [`vmul`][linalg.blas.vmul]
    """
    return {'mul': np.broadcast(a, b).size}

@visualisable(vmul_announce)
def vmul(a: ArrayLike, b: ArrayLike, *, progress: Progress) -> Any:
    """Return the elementwise product.
    
    Exactly like [`numpy.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html).
    
    Parameters
    ----------
    a, b : numpy.typing.ArrayLike
        Inputs.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    Any
        Product. An array for array like operands, a scalar otherwise.
    
    Complexity
    ----------
    For a resulting array with `n` elements there will be
    
    - `n` many scalar multiplications (`mul`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    - [`vmul_announce`][linalg.blas.vmul_announce]
    
    References
    ----------
    - [`numpy.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)
    """
    return ufunc_with_cb(np.multiply, a, b,
            cb=lambda *_: progress.update('mul'))

def vtruediv_announce(a: ArrayLike, b: ArrayLike) -> dict[str,int]:
    """`vtruediv` announcer.
    
    See also
    --------
    - [`vtruediv`][linalg.blas.vtruediv]
    """
    return {'truediv': np.broadcast(a, b).size}

@visualisable(vtruediv_announce)
def vtruediv(a: ArrayLike, b: ArrayLike, *, progress: Progress) -> Any:
    """Return the elementwise true quotient.
    
    Exactly like [`numpy.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html).
    
    Parameters
    ----------
    a, b : numpy.typing.ArrayLike
        Inputs.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    Any
        Floor quotient. An array for array like operands, a scalar otherwise.
    
    Complexity
    ----------
    For a resulting array with `n` elements there will be
    
    - `n` many scalar true divisions (`truediv`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    - [`vtruediv_announce`][linalg.blas.vtruediv_announce]
    
    References
    ----------
    - [`numpy.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html)
    """
    return ufunc_with_cb(np.divide, a, b,
            cb=lambda *_: progress.update('truediv'))

def vfloordiv_announce(a: ArrayLike, b: ArrayLike) -> dict[str,int]:
    """`vfloordiv` announcer.
    
    See also
    --------
    - [`vfloordiv`][linalg.blas.vfloordiv]
    """
    return {'floordiv': np.broadcast(a, b).size}

@visualisable(vfloordiv_announce)
def vfloordiv(a: ArrayLike, b: ArrayLike, *, progress: Progress) -> Any:
    """Return the elementwise floored quotient.
    
    Exactly like [`numpy.floor_divide`](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html).
    
    Parameters
    ----------
    a, b : numpy.typing.ArrayLike
        Inputs.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    Any
        True quotient. An array for array like operands, a scalar otherwise.
    
    Complexity
    ----------
    For a resulting array with `n` elements there will be
    
    - `n` many scalar floor divisions (`floordiv`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    - [`vfloordiv_announce`][linalg.blas.vfloordiv_announce]
    
    References
    ----------
    - [`numpy.floor_divide`](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html)
    """
    return ufunc_with_cb(np.floor_divide, a, b,
            cb=lambda *_: progress.update('floordiv'))

def vmod_announce(a: ArrayLike, b: ArrayLike) -> dict[str,int]:
    """`vmod` announcer.
    
    See also
    --------
    - [`vmod`][linalg.blas.vmod]
    """
    return {'mod': np.broadcast(a, b).size}

@visualisable(vmod_announce)
def vmod(a: ArrayLike, b: ArrayLike, *, progress: Progress) -> Any:
    """Return the elementwise remainder.
    
    Exactly like [`numpy.mod`](https://numpy.org/doc/stable/reference/generated/numpy.mod.html).
    
    Parameters
    ----------
    a, b : numpy.typing.ArrayLike
        Inputs.
    progress : Iterable[str]|bool|Progress = False
        Progress visualisation specification.
    
    Returns
    -------
    Any
        Remainder. An array for array like operands, a scalar otherwise.
    
    Complexity
    ----------
    For a resulting array with `n` elements there will be
    
    - `n` many scalar mod operations (`mod`).
    
    See also
    --------
    - [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
    - [`vmod_announce`][linalg.blas.vmod_announce]
    
    References
    ----------
    - [`numpy.mod`](https://numpy.org/doc/stable/reference/generated/numpy.mod.html)
    """
    return ufunc_with_cb(np.mod, a, b,
            cb=lambda *_: progress.update('mod'))
