"""Vectorial operations."""



import numpy as np
import numpy.typing as npt
from typing import Any
from collections.abc import Callable



__all__ = ('ufunc_with_cb', )



def _is_weak(x: Any) -> bool:
    """Return whether `x` is a NEP 50 weak scalar.
    
    - Python `int`, `float` or `complex` are weak
    - All others, `bool`, `numpy.int64`, ..., are strong
    
    References
    ----------
    - [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)
    """
    return (isinstance(x, (int, float, complex))
            and not isinstance(x, (bool, np.generic)))

def _weak_dtype(x: npt.ArrayLike) -> Any:
    """Return the `type` or `dtype` of `x`."""
    return type(x) if _is_weak(x) else np.asarray(x).dtype



def ufunc_with_cb(
    op:        np.ufunc,
    *operands: npt.ArrayLike,
    cb:        Callable[..., None] | None = None
) -> Any:
    """Apply a `numpy.ufunc` element-wise, with an optional callback.
    
    Same as calling `op` directly, but with an additional callback parameter.
    If set, gets called after every scalar operation with the scalar operands
    and the scalar results.
    
    The behaviour (throws, promotion, broadcasting, ...) should be exactly
    the same as calling `op` on its own.
    
    Tested for operations:
    
    - `numpy.positive`
    - `numpy.negative`
    - `numpy.add`
    - `numpy.subtract`
    - `numpy.multiply`
    - `numpy.divide`
    - `numpy.floor_divide`
    - `numpy.mod`
    - `numpy.divmod`
    - `numpy.modf`
    - `numpy.frexp`
    
    Improved code from the `numpy.nditer` examples.
    
    References
    ----------
    - [`numpy.nditer`](https://numpy.org/doc/stable/reference/generated/numpy.nditer.html)
    - [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)
    """
    if len(operands) > 1:
        dtypes = op.resolve_dtypes(tuple(_weak_dtype(o) for o in operands)
                                   + op.nout*(None,))
    else:
        dtypes = op.resolve_dtypes(tuple(np.asarray(o).dtype for o in operands)
                                   + op.nout*(None,))
    
    it = np.nditer(
        tuple(np.asarray(o, t if _is_weak(o) else None)
              for o, t in zip(operands, dtypes)) + op.nout*(None,),
        flags = ['buffered', 'refs_ok', 'zerosize_ok'],
        op_flags = op.nin * [['readonly']]
                   + op.nout * [['writeonly', 'allocate']],
        op_dtypes = dtypes,
        casting = 'unsafe'
    )
    with it:
        for args in it:
            op(*args[:op.nin], out=args[-op.nout:])
            if cb is not None:
                cb(*(a.item() for a in args))
        result = it.operands[-op.nout:]
        result = tuple(r if bool(r.ndim) else r[()] for r in result)
        return result[0] if op.nout==1 else result
