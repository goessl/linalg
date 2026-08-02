"""Progress visualisation.

Usage:

To visualise progress, just wrap the block that should be visualised into a
[`visualise`][linalg.progress.visualise] context.
All [`visualisable`][linalg.progress.visualisable] functions
will get a section with progress bars.
Nested visualisable functions increment the top most functions progress.

Implementation:

Decorate visualisable functions with
[`visualisable`][linalg.progress.visualisable]. It should announce
the number of scalar operations the function will perform.

[`notify`][linalg.progress.notify] when a scalar operation has happened
inside a visualisable function to publish the progress step.

Scalar helpers that perform the operation, notify and return the result
are provided for common Python operations.

For example:

```python
from itertools import zip_longest
from linalg.progress import visualisable, notify, visualise, mul
from typing import Any
from collections.abc import Sequence

@visualisable(lambda x, y: {'add':max(len(x), len(y))}) #announce total operations
def f(x:Sequence, y:Sequence) -> list:
    #Return vectorial `x + y`
    z = []
    for xi, yi in zip_longest(x, y, fillvalue=0):
        z.append(xi+yi)
        notify('add') #notify the visualisation to increment the add bar
    return z

@visualisable(lambda a, x, y: {'add':max(len(x), len(y)), 'mul':len(x)})
def g(a: Any, x: Sequence, y: Sequence) -> list:
    #Return vectorial `a*x+y`
    ax = []
    for xi in x:
        ax.append(mul(a, xi)) #operation and notification in one helper
    return f(ax, y) #g calls f internally: nesting is supported

x, y = [1, 2, 3], [4, 5, 6, 7]
f(x, y) #silent calculation

#visualise progress of additions and multiplications
#first for f, then for g
with visualise('add', 'mul'):
    f(x, y)
    g(2, x, y)
```

Will visualise:

```console
f
add: 100%|███████████████████████████████████████| 4/4 [00:20<00:00,  5.00s/it]
g
add:  50%|███████████████████▌                   | 2/4 [00:10<00:10,  5.00s/it]
mul:   0%|                                               | 0/3 [00:00<?, ?it/s]
```

Notes
-----
Mechanism.

Two `contextvars.ContextVar`s hold the state:

- `_requested_operations` - what the caller wants to see.
  Set by `visualise`, read by `visualisable`.
- `_active_progress_bars` - where notifications go.
  Set by the outermost visualisable functions.

| requested | active bars | meaning                            |
|-----------|-------------|------------------------------------|
| empty     | anything    | nothing requested to be visualised |
| non-empty | `None`      | user requested, you are top level  |
| non-empty | not `None`  | visualisation currently happening  |
"""



from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from tqdm.auto import tqdm
from typing import Any, ParamSpec, TypeVar
from collections.abc import Callable, Generator



__all__ = ('visualise', 'visualisable', 'notify',
           'pos', 'neg', 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod')



#visualisation framework
P = ParamSpec('P')
R = TypeVar('R')

_requested_operations: ContextVar[frozenset[str]] = ContextVar(
    'requested_operations',
    default=frozenset()
)
"""Operations that have been requested to be visualised."""

_active_progress_bars: ContextVar[dict[str,tqdm]|None] = ContextVar(
    'active_progress_bars',
    default=None
)
"""Currently printing progress bars."""

@contextmanager
def visualise(*operations:str) -> Generator[None]:
    """Request visualisation of functions within.
    
    Open as context (`with visualise(...):`) and all visualisable functions
    in this block will visualise their progress. Specify the operations you
    want to be visualised like `'add', 'mul', ...`.
    
    See also
    --------
    - [`visualisable`][linalg.progress.visualisable]
    - [`notify`][linalg.progress.notify]
    """
    token = _requested_operations.set(frozenset(operations))
    try:
        yield None
    finally:
        _requested_operations.reset(token)

def notify(operation:str, amount:int=1) -> None:
    """Increment corresponding progress bar, if it exists.
    
    Call in a [visualisable function][linalg.progress.visualisable]
    to publish the progress step(s).
    
    See also
    --------
    - [`visualise`][linalg.progress.visualise]
    - [`visualisable`][linalg.progress.visualisable]
    """
    pbars = _active_progress_bars.get()
    if pbars is not None and operation in pbars:
        pbars[operation].update(amount)

def visualisable(
    operations: Callable[P, dict[str, int]],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Make the function visualisable.
    
    Decorate a function that should be made visualisable.
    The decorator needs an additional function, that will receive the call
    arguments and should announce the total number of operations the function
    execution will [publish][linalg.progress.notify]
    including within subroutines.
    
    The number of operations that are announced
    at call time must match the total of notifications during execution.
    
    See also
    --------
    - [`visualise`][linalg.progress.visualise]
    - [`notify`][linalg.progress.notify]
    """
    def decorate(function:Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args:P.args, **kwargs:P.kwargs) -> R:
            requested = _requested_operations.get()
            #no progress requested
            #or already a deeper nested call
            # -> just run, no setup
            if not requested or _active_progress_bars.get() is not None:
                return function(*args, **kwargs)
            
            #non-empty requested & progress bars None -> setup bars
            selected = {op:n for op, n in operations(*args, **kwargs).items()
                             if op in requested}
            if selected: #title function section only if there will be bars
                tqdm.write(function.__name__)
            bars = { #create progress bars
                op: tqdm(total=n, desc=op, position=i)
                for i, (op, n) in enumerate(selected.items())
            } #even when no selected operations will happen
            #and no bars will be printed. Signal top level spot claimed
            
            token = _active_progress_bars.set(bars) #show them
            try:
                return function(*args, **kwargs)
            finally:
                _active_progress_bars.reset(token) #release top level spot
                for bar in bars.values(): #close them
                    bar.close()
        
        return wrapper
    return decorate



#scalar helper
def pos(a:Any) -> Any:
    """Return `+a` and notify for `pos`.
    
    See also
    --------
    - [`notify`][linalg.progress.notify]
    """
    r = +a
    notify('pos')
    return r

def neg(a:Any) -> Any:
    """Return `-a` and notify for `neg`.
    
    See also
    --------
    - [`notify`][linalg.progress.notify]
    """
    r = -a
    notify('neg')
    return r

def add(a:Any, b:Any) -> Any:
    """Return `a+b` and notify for `add`.
    
    See also
    --------
    - [`notify`][linalg.progress.notify]
    """
    r = a + b
    notify('add')
    return r

def sub(a:Any, b:Any) -> Any:
    """Return `a-b` and notify for `sub`.
    
    See also
    --------
    - [`notify`][linalg.progress.notify]
    """
    r = a - b
    notify('sub')
    return r

def mul(a:Any, b:Any) -> Any:
    """Return `a*b` and notify for `mul`.
    
    See also
    --------
    - [`notify`][linalg.progress.notify]
    """
    r = a * b
    notify('mul')
    return r

def truediv(a:Any, b:Any) -> Any:
    """Return `a/b` and notify for `truediv`.
    
    See also
    --------
    - [`notify`][linalg.progress.notify]
    """
    r = a / b
    notify('truediv')
    return r

def floordiv(a:Any, b:Any) -> Any:
    """Return `a//b` and notify for `floordiv`.
    
    See also
    --------
    - [`notify`][linalg.progress.notify]
    """
    r = a // b
    notify('floordiv')
    return r

def mod(a:Any, b:Any) -> Any:
    """Return `a%b` and notify for `mod`.
    
    See also
    --------
    - [`notify`][linalg.progress.notify]
    """
    r = a % b
    notify('mod')
    return r
